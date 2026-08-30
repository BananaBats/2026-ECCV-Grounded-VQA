#!/usr/bin/env python3
"""Independent pipeline-v6: Gemini noun phrase + bbox -> one SAM3 propagation.

Gemini sees the video before SAM3 starts. It selects the answer noun phrase, an
exact initialization frame, visibility boundaries, and explicit obj_id+bbox
targets. All boxes are registered in one SAM instance-tracker session before one
bidirectional propagation. No SAM output is ever sent back to Gemini and this
module imports no earlier pipeline implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    import torch
except ModuleNotFoundError:  # Plan-only runs do not need PyTorch or SAM3.
    torch = None  # type: ignore[assignment]


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MINSEON_ROOT = Path(
    os.environ.get("MINSEON_ROOT", str(REPO_ROOT))
).resolve()
PROJECT_ROOT = Path(
    os.environ.get("VQA_PROJECT_ROOT", str(REPO_ROOT))
).resolve()
SAM3_ROOT = Path(
    os.environ.get("SAM3_ROOT", str(REPO_ROOT / "third_party" / "sam3"))
).resolve()
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

DATA_SPLIT = os.environ.get("VQA_DATA_SPLIT", os.environ.get("DATA_SPLIT", "valid"))
VALID_ROOT = Path(
    os.environ.get("VQA_VALID_ROOT", str(PROJECT_ROOT / "dataset" / DATA_SPLIT))
).resolve()
VIDEO_ROOT = VALID_ROOT / "videos"
ANNOTATION_PATH = Path(
    os.environ.get(
        "VQA_ANNOTATION_PATH",
        str(VALID_ROOT / "annotations" / f"grounded_question_{DATA_SPLIT}.json"),
    )
).resolve()
DEFAULT_OUTPUT_ROOT = HERE / "outputs" / "bbox_one_shot"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_SAM3_CHECKPOINT = SAM3_ROOT / "checkpoints" / "sam3.pt"


def build_sam3_video_predictor(*args: Any, **kwargs: Any) -> Any:
    """Build SAM3 strictly from a local checkpoint, without Hugging Face calls."""
    if torch is None:
        raise RuntimeError(
            "SAM3 replay requires PyTorch. Plan-only mode does not; install the "
            "full replay environment before running without --plan-only."
        )
    from sam3.model_builder import (  # noqa: PLC0415
        build_sam3_video_predictor as _build_sam3_video_predictor,
    )

    checkpoint = Path(
        os.environ.get("SAM3_CHECKPOINT", str(DEFAULT_SAM3_CHECKPOINT))
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Local SAM3 checkpoint not found: {checkpoint}. "
            "Set SAM3_CHECKPOINT to a local sam3.pt file."
        )
    kwargs["checkpoint_path"] = str(checkpoint)
    return _build_sam3_video_predictor(*args, **kwargs)



class NounPhraseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_noun_phrase: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("answer_noun_phrase")
    @classmethod
    def validate_noun_phrase(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer_noun_phrase must not be empty")
        return value


class FrameSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_frame_idx: int = Field(ge=0)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class BoxTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obj_id: int = Field(ge=0)
    object_label: str
    box_2d: list[int]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("object_label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("object_label must not be empty")
        return value

    @field_validator("box_2d")
    @classmethod
    def validate_box(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError("box_2d must be [ymin, xmin, ymax, xmax]")
        if any(not 0 <= int(coordinate) <= 1000 for coordinate in value):
            raise ValueError("box_2d coordinates must be in 0..1000")
        ymin, xmin, ymax, xmax = map(int, value)
        if ymax <= ymin or xmax <= xmin:
            raise ValueError("box_2d must have positive width and height")
        return [ymin, xmin, ymax, xmax]


class BoxResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_summary: str
    first_visible_frame_idx: int = Field(ge=0)
    last_visible_frame_idx: int = Field(ge=0)
    visibility_boundary_reason: str
    objects: list[BoxTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_object_ids(self) -> "BoxResponse":
        obj_ids = [target.obj_id for target in self.objects]
        if len(obj_ids) != len(set(obj_ids)):
            raise ValueError("objects must have unique obj_id values")
        if sorted(obj_ids) != list(range(len(obj_ids))):
            raise ValueError("obj_id values must be contiguous integers starting at 0")
        return self


class SavedBoxPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    question_id: str
    question: str
    answer_noun_phrase: str
    target_frame_idx: int
    sam_target_frame_idx: int
    first_visible_frame_idx: int
    last_visible_frame_idx: int
    reasoning: str
    confidence: float
    frame_selection: dict[str, Any]
    visibility_boundary_reason: str
    decision_summary: str
    objects: list[BoxTarget]
    sam_stride: int
    boundary_radius: int
    source: str
    model: str
    call_metadata: dict[str, Any]


def save_json(value: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def save_pickle(value: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def save_rgb(image_rgb: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to save image: {path}")
    return path


def load_question(video_id: str, question_id: str) -> str:
    annotations = json.loads(ANNOTATION_PATH.read_text(encoding="utf-8"))
    video = annotations.get(video_id)
    if video is None:
        raise KeyError(f"{video_id!r} not found in {ANNOTATION_PATH}")
    for item in video.get("grounded_question", []):
        if str(item.get("id")) == str(question_id):
            question = str(item.get("question", "")).strip()
            if not question:
                raise ValueError(f"empty question for {video_id} q{question_id}")
            return question
    raise KeyError(f"question {question_id!r} not found for {video_id}")


def video_path_for(video_id: str) -> Path:
    exact = VIDEO_ROOT / f"{video_id}.mp4"
    if exact.exists():
        return exact
    matches = sorted(VIDEO_ROOT.glob(f"{video_id}.*"))
    if not matches:
        raise FileNotFoundError(f"video not found for {video_id} under {VIDEO_ROOT}")
    return matches[0]


def load_video_frames(video_path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    height, width = frames[0].shape[:2]
    return frames, {
        "num_frames": len(frames),
        "width": width,
        "height": height,
        "fps": fps,
    }


def probe_video(video_path: Path) -> dict[str, Any]:
    """Read video metadata without retaining decoded frames."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if num_frames < 1 or width < 1 or height < 1:
        raise RuntimeError(f"invalid video metadata for {video_path}")
    return {
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "fps": fps,
    }


def load_video_frames_at(
    video_path: Path, frame_indices: Sequence[int]
) -> dict[int, np.ndarray]:
    """Sequentially decode only requested RGB frames with bounded memory."""
    requested = sorted(set(map(int, frame_indices)))
    if not requested:
        return {}
    if requested[0] < 0:
        raise ValueError("frame indices must be non-negative")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    result: dict[int, np.ndarray] = {}
    wanted = set(requested)
    last_requested = requested[-1]
    try:
        frame_idx = 0
        while frame_idx <= last_requested:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx in wanted:
                result[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
    finally:
        capture.release()
    missing = [frame_idx for frame_idx in requested if frame_idx not in result]
    if missing:
        raise RuntimeError(f"could not decode requested frames {missing} from {video_path}")
    return result


def uniformly_sampled_indices(num_frames: int, max_frames: int) -> list[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))
    return sorted(
        set(np.linspace(0, num_frames - 1, max_frames, dtype=np.int64).tolist())
    )


def resize_for_proxy(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def build_planning_proxy(
    video_path: Path,
    frames: Sequence[np.ndarray],
    run_dir: Path,
    max_frames: int,
    max_dimension: int,
) -> tuple[Path, dict[str, Any]]:
    proxy_dir = run_dir / "gemini_planning_proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = proxy_dir / "planning_video.mp4"
    indices = uniformly_sampled_indices(len(frames), max_frames)
    first = resize_for_proxy(frames[indices[0]], max_dimension)
    height, width = first.shape[:2]
    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    proxy_fps = min(max(source_fps, 1.0), 6.0)
    writer = cv2.VideoWriter(
        str(proxy_path), cv2.VideoWriter_fourcc(*"mp4v"), proxy_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create planning proxy: {proxy_path}")
    try:
        for original_idx in indices:
            frame = resize_for_proxy(frames[original_idx], max_dimension)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            label = f"orig_frame={original_idx}"
            cv2.putText(
                bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(bgr)
    finally:
        writer.release()
    metadata = {
        "mode": "uniform_labeled_proxy",
        "original_video_path": str(video_path),
        "proxy_video_path": str(proxy_path),
        "original_frame_count": len(frames),
        "proxy_frame_count": len(indices),
        "max_frames": max_frames,
        "max_dimension": max_dimension,
        "proxy_size": [width, height],
        "source_fps": source_fps,
        "proxy_fps": proxy_fps,
        "proxy_to_original_frame": {
            str(proxy_idx): original_idx
            for proxy_idx, original_idx in enumerate(indices)
        },
    }
    save_json(metadata, proxy_dir / "frame_map.json")
    print(
        f"[Vertex AI] planning proxy: {len(frames)} original frames -> "
        f"{len(indices)} labeled frames at {width}x{height}",
        flush=True,
    )
    return proxy_path, metadata


def build_planning_proxy_streaming(
    video_path: Path,
    run_dir: Path,
    max_frames: int,
    max_dimension: int,
) -> tuple[Path, dict[str, Any]]:
    """Build the same labeled proxy while retaining only one decoded frame."""
    video_metadata = probe_video(video_path)
    indices = uniformly_sampled_indices(video_metadata["num_frames"], max_frames)
    first_rgb = load_video_frames_at(video_path, [indices[0]])[indices[0]]
    first = resize_for_proxy(first_rgb, max_dimension)
    height, width = first.shape[:2]
    proxy_dir = run_dir / "gemini_planning_proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = proxy_dir / "planning_video.mp4"
    proxy_fps = min(max(video_metadata["fps"], 1.0), 6.0)
    writer = cv2.VideoWriter(
        str(proxy_path), cv2.VideoWriter_fourcc(*"mp4v"), proxy_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create planning proxy: {proxy_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        writer.release()
        raise RuntimeError(f"could not open video: {video_path}")
    wanted = set(indices)
    try:
        frame_idx = 0
        written = 0
        while frame_idx <= indices[-1]:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_idx in wanted:
                frame = resize_for_proxy(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), max_dimension
                )
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                label = f"orig_frame={frame_idx}"
                cv2.putText(
                    bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 0), 4, cv2.LINE_AA,
                )
                cv2.putText(
                    bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                writer.write(bgr)
                written += 1
            frame_idx += 1
    finally:
        capture.release()
        writer.release()
    if written != len(indices):
        raise RuntimeError(
            f"proxy decode stopped early for {video_path}: {written}/{len(indices)}"
        )
    metadata = {
        "mode": "uniform_labeled_proxy",
        "original_video_path": str(video_path),
        "proxy_video_path": str(proxy_path),
        "original_frame_count": video_metadata["num_frames"],
        "proxy_frame_count": len(indices),
        "max_frames": max_frames,
        "max_dimension": max_dimension,
        "proxy_size": [width, height],
        "source_fps": video_metadata["fps"],
        "proxy_fps": proxy_fps,
        "proxy_to_original_frame": {
            str(proxy_idx): original_idx
            for proxy_idx, original_idx in enumerate(indices)
        },
    }
    save_json(metadata, proxy_dir / "frame_map.json")
    print(
        f"[Vertex AI] planning proxy: {video_metadata['num_frames']} original frames -> "
        f"{len(indices)} labeled frames at {width}x{height}",
        flush=True,
    )
    return proxy_path, metadata


def call_with_retry(
    label: str, callback: Callable[[], Any], max_attempts: int = 8
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[Gemini] {label} attempt={attempt}/{max_attempts}", flush=True)
            return callback()
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            transient = any(
                token in lowered
                for token in (
                    "408", "429", "500", "502", "503", "504", "timeout",
                    "temporarily unavailable", "too_many_requests",
                )
            )
            if not transient or attempt == max_attempts:
                raise
            match = re.search(r"(?:retry|retrying)\s+in\s+([0-9.]+)s", message, re.I)
            delay = float(match.group(1)) + 2 if match else min(60, 5 * 2 ** (attempt - 1))
            print(f"[Gemini] retrying in {delay:.1f}s: {exc}", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable Gemini retry state")


def function_config(name: str, schema: dict[str, Any]) -> types.GenerateContentConfig:
    declaration = types.FunctionDeclaration(
        name=name,
        description=f"Return the validated {name} result.",
        parameters_json_schema=schema,
    )
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=[name]
            )
        ),
        temperature=0.0,
    )


def call_function(
    client: genai.Client,
    model: str,
    name: str,
    schema: dict[str, Any],
    parts: list[types.Part],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = call_with_retry(
        name,
        lambda: client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=function_config(name, schema),
        ),
    )
    calls = response.function_calls or []
    if len(calls) != 1 or calls[0].name != name:
        returned = [call.name for call in calls]
        raise RuntimeError(f"expected one {name} call, got {returned}")
    usage = getattr(response, "usage_metadata", None)
    return dict(calls[0].args or {}), {
        "model": model,
        "function_name": name,
        "usage_metadata": usage.model_dump(mode="json") if usage else None,
    }


def request_noun_phrase(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    question: str,
) -> tuple[NounPhraseResponse, dict[str, Any]]:
    prompt = f"""
Inspect the complete labeled planning video and answer this tracking question:
{question}

Return exactly one concise visually trackable answer_noun_phrase identifying the
physical object or object set that directly answers the question. Include visible
attributes needed to distinguish it from similar objects. Do not select a frame or
return a bbox in this step. Do not answer with distractors or the manipulating
person. Call submit_answer_noun_phrase exactly once.
""".strip()
    arguments, metadata = call_function(
        client,
        model,
        "submit_answer_noun_phrase",
        NounPhraseResponse.model_json_schema(),
        [
            types.Part.from_bytes(data=proxy_path.read_bytes(), mime_type="video/mp4"),
            types.Part.from_text(text=prompt),
        ],
    )
    return NounPhraseResponse.model_validate(arguments), metadata


def request_target_frame(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    question: str,
    noun_phrase: str,
    allowed_frames: Sequence[int],
) -> tuple[FrameSelectionResponse, dict[str, Any]]:
    prompt = f"""
The complete-video answer target is authoritative: "{noun_phrase}".
Question: {question}

Choose one exact original-video frame for drawing a tight SAM3 initialization box
around every required physical instance. Prefer a frame where the target is
visible, separated, large, and minimally occluded. You MUST return one of these
SAM3 stride frames exactly:
{list(allowed_frames)}

Every proxy frame is labeled orig_frame=N. Do not estimate an in-between frame.
Call select_bbox_target_frame exactly once.
""".strip()
    arguments, metadata = call_function(
        client,
        model,
        "select_bbox_target_frame",
        FrameSelectionResponse.model_json_schema(),
        [
            types.Part.from_bytes(data=proxy_path.read_bytes(), mime_type="video/mp4"),
            types.Part.from_text(text=prompt),
        ],
    )
    selection = FrameSelectionResponse.model_validate(arguments)
    if selection.target_frame_idx not in set(allowed_frames):
        requested = selection.target_frame_idx
        corrected = nearest_frame(requested, allowed_frames)
        metadata.update(
            {
                "gemini_requested_frame_idx": requested,
                "python_corrected_frame_idx": corrected,
            }
        )
        selection = selection.model_copy(update={"target_frame_idx": corrected})
    return selection, metadata


def request_boxes(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    exact_frame_path: Path,
    question: str,
    noun_phrase: str,
    frame_idx: int,
    num_frames: int,
) -> tuple[BoxResponse, dict[str, Any]]:
    prompt = f"""
The complete-video answer target is authoritative: "{noun_phrase}".
Question: {question}

Draw one tight box around each distinct physical answer target visible in the
attached exact original frame {frame_idx}. Coordinates refer only to the exact
still image and use box_2d=[ymin,xmin,ymax,xmax] normalized to integer 0..1000.
Cover the whole physical object, never a union of multiple objects. Do not include
hands, occluders, shadows, or nearby distractors. Set every object_label to the
authoritative noun phrase verbatim. Only include sufficiently visible targets.
Assign each physical target a unique stable obj_id. Use contiguous integer IDs
starting at 0 (0, 1, ...). These IDs will be passed directly to the SAM instance
tracker, so never duplicate or skip an ID.

Also inspect the complete video and return inclusive original-video visibility
bounds in 0..{num_frames - 1}. first_visible_frame_idx is the earliest frame where
any required answer target is visibly present, including partial entry;
last_visible_frame_idx is the latest frame where any required target remains
visible, including partial exit. Briefly state the evidence in
visibility_boundary_reason.
Call submit_target_boxes exactly once.
""".strip()
    arguments, metadata = call_function(
        client,
        model,
        "submit_target_boxes",
        BoxResponse.model_json_schema(),
        [
            types.Part.from_bytes(data=proxy_path.read_bytes(), mime_type="video/mp4"),
            types.Part.from_text(text=f"Exact original frame {frame_idx}:"),
            types.Part.from_bytes(data=exact_frame_path.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
    )
    result = BoxResponse.model_validate(arguments)
    if not 0 <= result.first_visible_frame_idx <= result.last_visible_frame_idx < num_frames:
        raise ValueError("Gemini visibility bounds are invalid")
    return result, metadata


def stride_frames(num_frames: int, stride: int) -> list[int]:
    return list(range(0, num_frames, stride))


def boundary_dense_frames(
    num_frames: int,
    stride: int,
    first_visible: int,
    last_visible: int,
    radius: int,
) -> tuple[list[int], dict[str, Any]]:
    base = set(stride_frames(num_frames, stride))
    first_dense = set(
        range(max(0, first_visible - radius), min(num_frames - 1, first_visible + radius) + 1)
    )
    last_dense = set(
        range(max(0, last_visible - radius), min(num_frames - 1, last_visible + radius) + 1)
    )
    combined = sorted(base | first_dense | last_dense)
    return combined, {
        "sampling_mode": "stride_plus_gemini_visibility_boundaries",
        "sam_stride": stride,
        "boundary_radius": radius,
        "first_visible_frame_idx": first_visible,
        "last_visible_frame_idx": last_visible,
        "num_base_stride_frames": len(base),
        "num_additional_boundary_frames": len(combined) - len(base),
        "num_sam_frames": len(combined),
    }


def nearest_frame(requested: int, frames: Sequence[int]) -> int:
    return int(min(frames, key=lambda value: (abs(value - requested), value)))


def draw_box_plan(frame: np.ndarray, plan: SavedBoxPlan, path: Path) -> Path:
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    colors = [(240, 70, 70), (60, 190, 90), (70, 120, 245), (235, 180, 55)]
    for target in sorted(plan.objects, key=lambda item: item.obj_id):
        ymin, xmin, ymax, xmax = target.box_2d
        x1, y1 = round(xmin * width / 1000), round(ymin * height / 1000)
        x2, y2 = round(xmax * width / 1000), round(ymax * height / 1000)
        color = colors[target.obj_id % len(colors)]
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        cv2.putText(
            rendered, f"obj_id={target.obj_id} {target.object_label}",
            (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 2, cv2.LINE_AA,
        )
    return save_rgb(rendered, path)


def start_session(
    predictor: Any,
    frames: Sequence[np.ndarray],
    sampled_indices: Sequence[int],
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
) -> str:
    resource = [Image.fromarray(frames[index]) for index in sampled_indices]
    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": resource,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
    )
    return str(response["session_id"])


def box_to_tracker_corner_points(box_2d: Sequence[int]) -> list[list[float]]:
    """Encode a Gemini yxyx box as SAM tracker box-corner points.

    SAM's instance tracker represents a native box as its top-left and
    bottom-right corners with prompt labels 2 and 3.
    """
    ymin, xmin, ymax, xmax = map(float, box_2d)
    return [
        [xmin / 1000.0, ymin / 1000.0],
        [xmax / 1000.0, ymax / 1000.0],
    ]


def build_instance_box_prompt_request(
    session_id: str,
    sam_frame_idx: int,
    obj_id: int,
    box_2d: Sequence[int],
) -> dict[str, Any]:
    return {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": sam_frame_idx,
        "obj_id": int(obj_id),
        "points": box_to_tracker_corner_points(box_2d),
        "point_labels": [2, 3],
        "clear_old_points": True,
        "rel_coordinates": True,
    }


def propagate(
    predictor: Any, session_id: str, prompt_frame_idx: int
) -> dict[int, Any]:
    """Propagate from the explicit bbox prompt frame in both directions.

    Keep the same propagation contract as the earlier point pipeline. Supplying the prompt
    frame explicitly is important for tracker prompts, and ``both`` preserves
    predictions before as well as after the initialization frame.
    """
    outputs: dict[int, Any] = {}
    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "both",
        "start_frame_index": int(prompt_frame_idx),
    }
    for response in predictor.handle_stream_request(request=request):
        outputs[int(response["frame_index"])] = response["outputs"]
    return outputs


def remap_outputs(
    sampled_outputs: dict[int, Any], sampled_indices: Sequence[int]
) -> dict[int, Any]:
    return {
        int(sampled_indices[int(sampled_idx)]): output
        for sampled_idx, output in sampled_outputs.items()
    }


def output_obj_ids(frame_output: Any) -> list[int]:
    if not isinstance(frame_output, dict):
        return []
    source = frame_output.get("outputs", frame_output)
    return [
        int(value)
        for value in to_numpy(source.get("out_obj_ids", [])).reshape(-1)
    ]


def run_instance_bbox_session(
    predictor: Any,
    frames: Sequence[np.ndarray],
    sampled_indices: Sequence[int],
    plan: SavedBoxPlan,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    session_id = start_session(
        predictor,
        frames,
        sampled_indices,
        offload_video_to_cpu,
        offload_state_to_cpu,
    )
    try:
        predictor.handle_request(
            request={"type": "reset_session", "session_id": session_id}
        )
        for target in sorted(plan.objects, key=lambda item: item.obj_id):
            request = build_instance_box_prompt_request(
                session_id,
                plan.sam_target_frame_idx,
                target.obj_id,
                target.box_2d,
            )
            prompt_response = predictor.handle_request(request=request)
            prompt_obj_ids = output_obj_ids(prompt_response.get("outputs"))
            if target.obj_id not in prompt_obj_ids:
                raise RuntimeError(
                    f"SAM tracker prompt output omitted obj_id={target.obj_id}; "
                    f"returned IDs: {prompt_obj_ids}"
                )
            logs.append(
                {
                    "status": "ok",
                    "obj_id": target.obj_id,
                    "prompt_method": "instance_tracker_native_box",
                    "object_label": target.object_label,
                    "box_2d": target.box_2d,
                    "tracker_box_corner_points": request["points"],
                    "tracker_box_labels": request["point_labels"],
                    "sam_frame_idx": plan.sam_target_frame_idx,
                    "response_success": bool(prompt_response.get("is_success", True)),
                    "prompt_output_obj_ids": prompt_obj_ids,
                }
            )
        raw = remap_outputs(
            propagate(predictor, session_id, plan.sam_target_frame_idx),
            sampled_indices,
        )
        return raw, logs
    finally:
        close_session(predictor, session_id)


def collect_object_ids(raw: dict[int, Any]) -> list[int]:
    result: set[int] = set()
    for output in raw.values():
        source = output.get("outputs", output) if isinstance(output, dict) else {}
        for value in to_numpy(source.get("out_obj_ids", [])).reshape(-1):
            result.add(int(value))
    return sorted(result)


def summarize(raw: dict[int, Any], num_video_frames: int) -> dict[str, Any]:
    object_frames: dict[int, list[int]] = {}
    for frame_idx, output in raw.items():
        source = output.get("outputs", output) if isinstance(output, dict) else {}
        ids = to_numpy(source.get("out_obj_ids", [])).reshape(-1)
        masks = to_numpy(source.get("out_binary_masks", []))
        if masks.ndim == 2:
            masks = masks[None, ...]
        for index, obj_id in enumerate(ids):
            if index < len(masks) and np.any(masks[index]):
                object_frames.setdefault(int(obj_id), []).append(int(frame_idx))
    non_empty = sorted({frame for values in object_frames.values() for frame in values})
    return {
        "num_video_frames": num_video_frames,
        "num_output_frames": len(raw),
        "num_non_empty_frames": len(non_empty),
        "first_non_empty_frame": non_empty[0] if non_empty else None,
        "last_non_empty_frame": non_empty[-1] if non_empty else None,
        "object_ids": sorted(object_frames),
        "objects": {
            str(obj_id): {
                "num_non_empty_frames": len(frames),
                "first_frame": min(frames),
                "last_frame": max(frames),
            }
            for obj_id, frames in sorted(object_frames.items())
        },
        "non_empty_frames": non_empty,
    }


def to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def output_boxes_for_visualization(
    output: dict[str, Any], frame_height: int, frame_width: int
) -> list[tuple[int, int, int, int, int, float | None]]:
    """Convert SAM3 normalized top-left XYWH outputs to clipped pixel boxes."""
    obj_ids = to_numpy(output.get("out_obj_ids", [])).reshape(-1)
    boxes = to_numpy(output.get("out_boxes_xywh", []))
    probs = to_numpy(output.get("out_probs", [])).reshape(-1)
    if boxes.size == 0:
        return []
    boxes = boxes.reshape(-1, 4)
    count = min(len(obj_ids), len(boxes))
    rendered_boxes: list[tuple[int, int, int, int, int, float | None]] = []
    for index in range(count):
        x, y, width, height = map(float, boxes[index])
        x1 = int(round(x * frame_width))
        y1 = int(round(y * frame_height))
        x2 = int(round((x + width) * frame_width))
        y2 = int(round((y + height) * frame_height))
        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width - 1, x2))
        y2 = max(0, min(frame_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        probability = float(probs[index]) if index < len(probs) else None
        rendered_boxes.append(
            (int(obj_ids[index]), x1, y1, x2, y2, probability)
        )
    return rendered_boxes


def render_visualizations(
    frames: Sequence[np.ndarray], raw: dict[int, Any], output_dir: Path, stride: int,
    extra_indices: Sequence[int],
) -> list[str]:
    indices = set(range(0, len(frames), max(1, stride)))
    indices.update(extra_indices)
    indices.add(len(frames) - 1)
    paths: list[str] = []
    palette = [(240, 70, 70), (60, 190, 90), (70, 120, 245), (235, 180, 55)]
    for frame_idx in sorted(index for index in indices if 0 <= index < len(frames)):
        rendered = frames[frame_idx].copy()
        frame_height, frame_width = rendered.shape[:2]
        for obj_id, x1, y1, x2, y2, probability in output_boxes_for_visualization(
            raw.get(frame_idx, {}), frame_height, frame_width
        ):
            color = palette[obj_id % len(palette)]
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            label = f"obj_id={obj_id}"
            if probability is not None:
                label += f" p={probability:.2f}"
            cv2.putText(
                rendered,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            rendered, f"frame={frame_idx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), 2, cv2.LINE_AA,
        )
        path = save_rgb(rendered, output_dir / f"frame_{frame_idx:05d}.png")
        paths.append(str(path))
    return paths


def close_session(predictor: Any, session_id: str) -> None:
    try:
        predictor.handle_request(
            request={"type": "close_session", "session_id": session_id}
        )
    except Exception as exc:
        print(f"[SAM3] close warning: {exc}", flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sam_stride < 1:
        raise ValueError("--sam-stride must be at least 1")
    if args.boundary_radius < 0:
        raise ValueError("--boundary-radius must be non-negative")
    video_id, question_id = str(args.video_id), str(args.question_id)
    run_dir = Path(args.output_root) / f"{video_id}_q{question_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_path_for(video_id)
    question = load_question(video_id, question_id)
    print(f"[PipelineV6] loading {video_id} q{question_id}", flush=True)
    frames, video_info = load_video_frames(video_path)
    proxy_path, proxy_metadata = build_planning_proxy(
        video_path,
        frames,
        run_dir,
        args.gemini_video_max_frames,
        args.gemini_video_max_dimension,
    )
    client = genai.Client(
        vertexai=True,
        project=args.google_cloud_project,
        location=args.google_cloud_location,
        http_options=types.HttpOptions(api_version="v1", timeout=300_000),
    )
    noun, noun_meta = request_noun_phrase(
        client, args.gemini_model, proxy_path, question
    )
    base_stride_frames = stride_frames(len(frames), args.sam_stride)
    frame_selection, frame_meta = request_target_frame(
        client,
        args.gemini_model,
        proxy_path,
        question,
        noun.answer_noun_phrase,
        base_stride_frames,
    )
    if args.target_frame != "auto":
        frame_selection = frame_selection.model_copy(
            update={
                "target_frame_idx": nearest_frame(
                    int(args.target_frame), base_stride_frames
                )
            }
        )
    target_frame_idx = frame_selection.target_frame_idx
    exact_frame_path = save_rgb(frames[target_frame_idx], run_dir / "target_frame.png")
    boxes, boxes_meta = request_boxes(
        client,
        args.gemini_model,
        proxy_path,
        exact_frame_path,
        question,
        noun.answer_noun_phrase,
        target_frame_idx,
        len(frames),
    )
    sampled_indices, sampling = boundary_dense_frames(
        len(frames),
        args.sam_stride,
        boxes.first_visible_frame_idx,
        boxes.last_visible_frame_idx,
        args.boundary_radius,
    )
    sam_target_frame_idx = sampled_indices.index(target_frame_idx)
    plan = SavedBoxPlan(
        video_id=video_id,
        question_id=question_id,
        question=question,
        answer_noun_phrase=noun.answer_noun_phrase,
        target_frame_idx=target_frame_idx,
        sam_target_frame_idx=sam_target_frame_idx,
        first_visible_frame_idx=boxes.first_visible_frame_idx,
        last_visible_frame_idx=boxes.last_visible_frame_idx,
        reasoning=noun.reasoning,
        confidence=noun.confidence,
        frame_selection={
            **frame_selection.model_dump(mode="json"),
            "call_metadata": frame_meta,
        },
        visibility_boundary_reason=boxes.visibility_boundary_reason,
        decision_summary=boxes.decision_summary,
        objects=boxes.objects,
        sam_stride=args.sam_stride,
        boundary_radius=args.boundary_radius,
        source="gemini_noun_phrase_then_aligned_frame_then_instance_tracker_boxes",
        model=args.gemini_model,
        call_metadata={"noun_phrase": noun_meta, "frame": frame_meta, "boxes": boxes_meta},
    )
    save_json(plan.model_dump(mode="json"), run_dir / "gemini_bbox_plan.json")
    save_json(
        {
            "answer_noun_phrase": plan.answer_noun_phrase,
            "question": question,
            "reasoning": noun.reasoning,
            "confidence": noun.confidence,
            "source": "pipeline-v6-separated-noun-phrase-call",
            "call_metadata": noun_meta,
        },
        run_dir / "gemini_answer_prompt.json",
    )
    draw_box_plan(frames[target_frame_idx], plan, run_dir / "gemini_bbox_overlay.png")
    save_json(
        {
            **sampling,
            "num_original_frames": len(frames),
            "target_frame_idx": target_frame_idx,
            "sam_target_frame_idx": sam_target_frame_idx,
            "sam_to_original_frame": {
                str(index): original for index, original in enumerate(sampled_indices)
            },
            "gemini_planning_proxy": proxy_metadata,
        },
        run_dir / "sam_frame_map.json",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video inference requires CUDA")
    predictor = build_sam3_video_predictor(
        gpus_to_use=[torch.cuda.current_device()]
    )
    print(
        f"[SAM3] registering {len(plan.objects)} Gemini obj_id+bbox prompt(s) "
        "in one instance-tracker session, then propagating once",
        flush=True,
    )
    raw, prompt_log = run_instance_bbox_session(
        predictor,
        frames,
        sampled_indices,
        plan,
        args.offload_video_to_cpu,
        args.offload_state_to_cpu,
    )
    save_json(prompt_log, run_dir / "bbox_prompt_execution_log.json")

    baseline_path = save_pickle(raw, run_dir / "baseline_raw_outputs_true.pkl")
    refined_path = save_pickle(raw, run_dir / "refined_raw_outputs_true.pkl")
    tracking = summarize(raw, len(frames))
    tracking.update(
        {
            "video_id": video_id,
            "question_id": question_id,
            "question": question,
            "initialization": "gemini_instance_tracker_boxes",
            "num_input_boxes": len(plan.objects),
            "target_frame_idx": target_frame_idx,
            "sam_target_frame_idx": sam_target_frame_idx,
            "first_visible_frame_idx": plan.first_visible_frame_idx,
            "last_visible_frame_idx": plan.last_visible_frame_idx,
            "boundary_radius": args.boundary_radius,
            "num_sam_frames": len(sampled_indices),
            "raw_output_path": str(baseline_path),
        }
    )
    tracking["visualization_paths"] = render_visualizations(
        frames,
        raw,
        run_dir / "baseline_visualizations",
        args.vis_stride,
        [target_frame_idx, plan.first_visible_frame_idx, plan.last_visible_frame_idx],
    )
    save_json(tracking, run_dir / "baseline_summary.json")
    refined_summary = {
        **tracking,
        "raw_output_path": str(refined_path),
        "pipeline_variant": "pipeline-v6-bbox-one-shot",
        "refinement_policy": "disabled",
        "gemini_sam_interaction": False,
        "num_refinement_iterations": 0,
        "effective_actions": [],
    }
    save_json(refined_summary, run_dir / "refined_summary.json")
    save_json(
        {
            "pipeline_variant": "pipeline-v6-bbox-one-shot",
            "policy": "no Gemini/SAM interaction after bbox initialization",
            "effective_actions": [],
            "iterations": [],
        },
        run_dir / "action_execution_log.json",
    )
    comparison = {
        "video_id": video_id,
        "question_id": question_id,
        "question": question,
        "initialization": "gemini_instance_tracker_boxes",
        "sam_stride": args.sam_stride,
        "target_frame_idx": target_frame_idx,
        "sam_target_frame_idx": sam_target_frame_idx,
        "first_visible_frame_idx": plan.first_visible_frame_idx,
        "last_visible_frame_idx": plan.last_visible_frame_idx,
        "boundary_radius": args.boundary_radius,
        "num_sam_frames": len(sampled_indices),
        "num_initial_boxes": len(plan.objects),
        "object_ids": collect_object_ids(raw),
        "pipeline_variant": "pipeline-v6-bbox-one-shot",
        "refinement_policy": "disabled",
        "gemini_sam_interaction": False,
        "num_refinement_iterations": 0,
        "effective_action_count": 0,
    }
    save_json(comparison, run_dir / "comparison_summary.json")
    print(f"[PipelineV6] complete: {run_dir}", flush=True)
    return comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    parser = argparse.ArgumentParser(
        description="Independent Gemini bbox -> one-shot SAM3 pipeline"
    )
    parser.add_argument("--video-id", "--video_id", dest="video_id", required=True)
    parser.add_argument("--question-id", "--question_id", dest="question_id", required=True)
    parser.add_argument("--sam-stride", "--sam_stride", dest="sam_stride", type=int, default=10)
    parser.add_argument("--boundary-radius", "--boundary_radius", dest="boundary_radius", type=int, default=9)
    parser.add_argument("--target-frame", "--target_frame", dest="target_frame", default="auto")
    parser.add_argument("--force-gemini", action="store_true", help="retained for CLI compatibility; bbox plans are always regenerated")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--vis-stride", type=int, default=60)
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--gemini-model", "--gemini_model", "--tuned-model-endpoint", dest="gemini_model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--gemini-video-max-frames", type=int, default=192)
    parser.add_argument("--gemini-video-max-dimension", type=int, default=512)
    parser.add_argument("--google-cloud-project", "--google_cloud_project", dest="google_cloud_project", default=project, required=not project)
    parser.add_argument("--google-cloud-location", "--google_cloud_location", dest="google_cloud_location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--output-root", "--output_root", dest="output_root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


def main() -> None:
    print(
        "[PipelineV6] Gemini noun phrase + exact-frame bbox -> one SAM3 propagation",
        flush=True,
    )
    run(parse_args())


if __name__ == "__main__":
    main()
