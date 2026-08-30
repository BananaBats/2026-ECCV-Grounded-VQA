#!/usr/bin/env python3
"""Multi-anchor Gemini bbox -> SAM3 tracking sessions for pipeline-v7.

This is intentionally independent from pipeline-v6 outputs.  It reuses only the
local SAM3/Vertex utility functions from ``pipeline-v6/box_baseline.py``.

Policy contract:
* Gemini first tries to select one stride-aligned frame containing every target.
* If that is impossible, it selects the smallest useful set of anchor frames.
* Every anchor stores the visible subset of a single global, stable obj_id set.
* An RL policy can replace Gemini's anchor-selection JSON through
  ``--anchor-selection-json``; the tracker and evaluator then remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


HERE = Path(__file__).resolve().parent
V6_ROOT = Path(os.environ.get("V6_ROOT", str(HERE))).resolve()
if str(V6_ROOT) not in sys.path:
    sys.path.insert(0, str(V6_ROOT))

import box_baseline as base  # noqa: E402


DEFAULT_OUTPUT_ROOT = HERE / "outputs" / "multi_anchor_box_baseline"


class VisibilitySegment(BaseModel):
    """One contiguous appearance interval of a target in the original video."""

    model_config = ConfigDict(extra="forbid")

    first_frame_idx: int = Field(ge=0)
    last_frame_idx: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> "VisibilitySegment":
        if self.first_frame_idx > self.last_frame_idx:
            raise ValueError("segment first_frame_idx must not exceed last_frame_idx")
        return self


class TargetIdentity(BaseModel):
    """A target identity shared across every anchor frame."""

    model_config = ConfigDict(extra="forbid")

    obj_id: int = Field(ge=0)
    object_label: str
    visual_identity_description: str
    visibility_segments: list[VisibilitySegment] = Field(default_factory=list)

    @field_validator("object_label", "visual_identity_description")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target identity fields must not be empty")
        return value

    @field_validator("visibility_segments")
    @classmethod
    def sort_segments(cls, value: list[VisibilitySegment]) -> list[VisibilitySegment]:
        return sorted(value, key=lambda item: item.first_frame_idx)


class AnchorSelection(BaseModel):
    """A single action chosen by Gemini now and by the RL policy later."""

    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    obj_ids: list[int] = Field(min_length=1)
    reason: str

    @field_validator("obj_ids")
    @classmethod
    def unique_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("anchor obj_ids must be unique")
        return [int(item) for item in value]


class AnchorSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_summary: str
    first_visible_frame_idx: int = Field(ge=0)
    last_visible_frame_idx: int = Field(ge=0)
    visibility_boundary_reason: str
    targets: list[TargetIdentity] = Field(min_length=1)
    anchors: list[AnchorSelection] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_global_identity_and_coverage(self) -> "AnchorSelectionResponse":
        target_ids = [target.obj_id for target in self.targets]
        if sorted(target_ids) != list(range(len(target_ids))):
            raise ValueError("targets must use contiguous obj_ids starting at 0")
        if self.first_visible_frame_idx > self.last_visible_frame_idx:
            raise ValueError("first_visible_frame_idx must not exceed last_visible_frame_idx")
        anchor_frames = [anchor.frame_idx for anchor in self.anchors]
        if len(anchor_frames) != len(set(anchor_frames)):
            raise ValueError("anchor frame_idx values must be unique")
        known_ids = set(target_ids)
        covered_ids = {obj_id for anchor in self.anchors for obj_id in anchor.obj_ids}
        if not covered_ids <= known_ids:
            raise ValueError("anchors refer to an unknown obj_id")
        if covered_ids != known_ids:
            raise ValueError("every target obj_id must be covered by at least one anchor")
        return self


class AnchorBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obj_id: int = Field(ge=0)
    box_2d: list[int]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("box_2d")
    @classmethod
    def valid_box(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError("box_2d must be [ymin, xmin, ymax, xmax]")
        if any(not 0 <= int(item) <= 1000 for item in value):
            raise ValueError("box_2d values must be in 0..1000")
        ymin, xmin, ymax, xmax = map(int, value)
        if ymax <= ymin or xmax <= xmin:
            raise ValueError("box_2d must have positive area")
        return [ymin, xmin, ymax, xmax]


class AnchorBoxesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boxes: list[AnchorBox] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_obj_ids(self) -> "AnchorBoxesResponse":
        ids = [box.obj_id for box in self.boxes]
        if len(ids) != len(set(ids)):
            raise ValueError("a frame can contain at most one bbox per obj_id")
        return self


class SavedAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int
    sam_frame_idx: int
    obj_ids: list[int]
    reason: str
    boxes: list[AnchorBox]
    exact_frame_path: str
    call_metadata: dict[str, Any]


class SavedMultiAnchorPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    question_id: str
    question: str
    answer_noun_phrase: str
    noun_reasoning: str
    noun_confidence: float
    decision_summary: str
    first_visible_frame_idx: int
    last_visible_frame_idx: int
    visibility_boundary_reason: str
    targets: list[TargetIdentity]
    anchors: list[SavedAnchor]
    sam_stride: int
    sam_target_fps: float | None = None
    required_frame_stride: int | None = None
    boundary_radius: int
    max_anchor_frames: int
    anchor_policy: str
    model: str
    gemini_video_fps: float
    call_metadata: dict[str, Any]


NOUN_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "wrong"]},
        "corrected_noun_phrase": {"type": "string"},
        "evidence_frames": {"type": "array", "items": {"type": "integer"}},
        "expected_instance_count": {"type": "integer", "minimum": 1},
    },
    "required": ["reasoning", "verdict", "expected_instance_count"],
}


def proxy_video_part(proxy_path: Path, fps: float) -> types.Part:
    """Video part with an explicit Gemini sampling rate (default API rate is 1fps)."""
    part = types.Part.from_bytes(data=proxy_path.read_bytes(), mime_type="video/mp4")
    part.video_metadata = types.VideoMetadata(fps=fps)
    return part


def request_noun_phrase(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    question: str,
    video_fps: float,
) -> tuple[Any, dict[str, Any]]:
    prompt = f"""
Inspect the complete labeled planning video and answer this tracking question:
{question}

Return exactly one concise visually trackable answer_noun_phrase identifying the
physical object or object set that directly answers the question. Include visible
attributes needed to distinguish it from similar objects. Do not select a frame or
return a bbox in this step. Do not answer with distractors or the manipulating
person. Call submit_answer_noun_phrase exactly once.
""".strip()
    arguments, metadata = base.call_function(
        client,
        model,
        "submit_answer_noun_phrase",
        base.NounPhraseResponse.model_json_schema(),
        [
            proxy_video_part(proxy_path, video_fps),
            types.Part.from_text(text=prompt),
        ],
    )
    return base.NounPhraseResponse.model_validate(arguments), metadata


def verify_noun_phrase(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    question: str,
    candidate: str,
    video_fps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Critic pass: re-derive the answer from scratch and judge the candidate.

    Probe results (2026-07-24): corrected 3/6 known planning failures with zero
    false alarms on controls; best-of-N voting only fixed a subset of these, so
    a single temperature-0 critic call is the better trade.
    """
    prompt = f"""
A previous analyst proposed this answer target for the tracking question below.
Candidate answer: "{candidate}"
Question: {question}

Re-derive the answer from scratch. Reason step by step about what the question
asks for, including any counterfactual or physical reasoning (e.g. "which
object WOULD it collide with", "what stops the motion", "which object was
removed"). Decide which physical object in the video satisfies the question,
then compare with the candidate.
Rules: the answer must be a visually trackable physical object; it must never
be a person, hand, arm, or body part; distinguish the acting object from the
object acted upon.
Also report expected_instance_count: how many distinct physical object
instances the question requires tracking (e.g. three separate letters = 3).
If the candidate is right, verdict="correct". If not, verdict="wrong" and give
corrected_noun_phrase plus the orig_frame label numbers that support it.
Call submit_answer_verification exactly once.
""".strip()
    arguments, metadata = base.call_function(
        client,
        model,
        "submit_answer_verification",
        NOUN_VERIFICATION_SCHEMA,
        [
            proxy_video_part(proxy_path, video_fps),
            types.Part.from_text(text=prompt),
        ],
    )
    return arguments, metadata


def request_anchor_selection(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    question: str,
    noun_phrase: str,
    allowed_frames: Sequence[int],
    max_anchor_frames: int,
    video_fps: float,
    instance_count_hint: int | None = None,
) -> tuple[AnchorSelectionResponse, dict[str, Any]]:
    hint_text = (
        f"\nThe answer target comprises {instance_count_hint} distinct physical "
        "instances; give each its own obj_id."
        if instance_count_hint and instance_count_hint > 1
        else ""
    )
    prompt = f"""
The complete-video answer target is authoritative: "{noun_phrase}".
Question: {question}

Identify every distinct physical answer instance and give each a stable obj_id
(0, 1, ...).  If the answer denotes multiple separate instances (e.g. several
letters, cups, or shapes), you MUST create one target per instance; never merge
them into a single target.{hint_text}  Then choose anchor frames for SAM3 bbox
prompts.

First try to choose ONE frame in which every target instance is visible,
separated, and minimally occluded.  Only if no such frame exists, choose the
smallest set of up to {max_anchor_frames} frames that covers all target
instances.  In each anchor's obj_ids, include only the target instances visibly
present in that exact frame.  Do not add visually similar distractors, hands,
scene regions, or objects not required by the question.

Every anchor frame MUST be one of these original-video stride frames:
{list(allowed_frames)}

The proxy frames are labeled orig_frame=N.  Also return inclusive visibility
bounds for the target set over the original video.  For EACH target, return
visibility_segments: every contiguous interval of original-video frames in which
that target is visible, as inclusive first_frame_idx/last_frame_idx pairs.  A
target shown, removed, and shown again has multiple segments.  If a target stays
visible until the video ends, end its final segment at the last original frame
index.  Call submit_multi_anchor_selection exactly once.
""".strip()
    arguments, metadata = base.call_function(
        client,
        model,
        "submit_multi_anchor_selection",
        AnchorSelectionResponse.model_json_schema(),
        [
            proxy_video_part(proxy_path, video_fps),
            types.Part.from_text(text=prompt),
        ],
    )
    selection = AnchorSelectionResponse.model_validate(arguments)
    return align_anchor_selection(selection, allowed_frames, max_anchor_frames, metadata)


def align_anchor_selection(
    selection: AnchorSelectionResponse,
    allowed_frames: Sequence[int],
    max_anchor_frames: int,
    metadata: dict[str, Any],
) -> tuple[AnchorSelectionResponse, dict[str, Any]]:
    """Snap proxy mistakes to stride frames while preserving target coverage."""
    if len(selection.anchors) > max_anchor_frames:
        raise ValueError(
            f"planner chose {len(selection.anchors)} anchors; max is {max_anchor_frames}"
        )
    allowed = set(map(int, allowed_frames))
    aligned: dict[int, AnchorSelection] = {}
    corrections: list[dict[str, int]] = []
    for anchor in selection.anchors:
        frame_idx = int(anchor.frame_idx)
        if frame_idx not in allowed:
            corrected = base.nearest_frame(frame_idx, allowed_frames)
            corrections.append({"requested_frame_idx": frame_idx, "aligned_frame_idx": corrected})
            frame_idx = corrected
        previous = aligned.get(frame_idx)
        if previous is None:
            aligned[frame_idx] = anchor.model_copy(update={"frame_idx": frame_idx})
        else:
            merged_ids = sorted(set(previous.obj_ids) | set(anchor.obj_ids))
            aligned[frame_idx] = previous.model_copy(update={"obj_ids": merged_ids})
    aligned_selection = selection.model_copy(
        update={"anchors": [aligned[index] for index in sorted(aligned)]}
    )
    aligned_selection = AnchorSelectionResponse.model_validate(
        aligned_selection.model_dump(mode="json")
    )
    if len(aligned_selection.anchors) > max_anchor_frames:
        raise ValueError("frame alignment exceeded max_anchor_frames")
    if corrections:
        metadata = {**metadata, "frame_alignment": corrections}
    return aligned_selection, metadata


def load_anchor_selection(
    path: Path,
    allowed_frames: Sequence[int],
    max_anchor_frames: int,
) -> tuple[AnchorSelectionResponse, dict[str, Any]]:
    selection = AnchorSelectionResponse.model_validate_json(path.read_text(encoding="utf-8"))
    aligned, metadata = align_anchor_selection(
        selection,
        allowed_frames,
        max_anchor_frames,
        {"source": "anchor_selection_json", "path": str(path.resolve())},
    )
    return aligned, metadata


def request_anchor_boxes(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    exact_frame_path: Path,
    question: str,
    noun_phrase: str,
    anchor: AnchorSelection,
    identities: Sequence[TargetIdentity],
    video_fps: float,
) -> tuple[AnchorBoxesResponse, dict[str, Any]]:
    identity_by_id = {item.obj_id: item for item in identities}
    requested = [identity_by_id[obj_id] for obj_id in anchor.obj_ids]
    identity_text = "\n".join(
        f"- obj_id={item.obj_id}: {item.object_label}; {item.visual_identity_description}"
        for item in requested
    )
    prompt = f"""
The answer target is "{noun_phrase}" for this question:
{question}

This exact original image is frame {anchor.frame_idx}.  Draw a tight normalized
box_2d=[ymin,xmin,ymax,xmax] for EVERY and ONLY the requested target identities:
{identity_text}

Return exactly these obj_ids: {anchor.obj_ids}.  Use the supplied stable obj_id
unchanged.  Do not include hands, occluders, shadows, scene regions, or extra
instances.  Each box must cover the full visible physical object in this exact
image.  Call submit_multi_anchor_boxes exactly once.
""".strip()
    arguments, metadata = base.call_function(
        client,
        model,
        "submit_multi_anchor_boxes",
        AnchorBoxesResponse.model_json_schema(),
        [
            proxy_video_part(proxy_path, video_fps),
            types.Part.from_text(text=f"Exact original frame {anchor.frame_idx}:"),
            types.Part.from_bytes(data=exact_frame_path.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
    )
    result = AnchorBoxesResponse.model_validate(arguments)
    actual_ids = sorted(box.obj_id for box in result.boxes)
    expected_ids = sorted(anchor.obj_ids)
    if actual_ids != expected_ids:
        raise ValueError(
            f"anchor frame {anchor.frame_idx} returned obj_ids={actual_ids}; "
            f"expected {expected_ids}"
        )
    return result, metadata


def draw_anchor_plan(
    frames: Sequence[Any], plan: SavedMultiAnchorPlan, output_dir: Path) -> list[str]:
    paths: list[str] = []
    for anchor in plan.anchors:
        rendered = frames[anchor.frame_idx].copy()
        height, width = rendered.shape[:2]
        for box in anchor.boxes:
            ymin, xmin, ymax, xmax = box.box_2d
            x1, y1 = round(xmin * width / 1000), round(ymin * height / 1000)
            x2, y2 = round(xmax * width / 1000), round(ymax * height / 1000)
            color = [(240, 70, 70), (60, 190, 90), (70, 120, 245), (235, 180, 55)][box.obj_id % 4]
            base.cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 3, base.cv2.LINE_AA)
            base.cv2.putText(
                rendered,
                f"anchor obj_id={box.obj_id}",
                (x1, max(22, y1 - 7)),
                base.cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                base.cv2.LINE_AA,
            )
        path = base.save_rgb(
            rendered,
            output_dir / f"anchor_{anchor.frame_idx:05d}.png",
        )
        paths.append(str(path))
    return paths


def run_multi_anchor_session(
    predictor: Any,
    frames: Sequence[Any],
    sampled_indices: Sequence[int],
    plan: SavedMultiAnchorPlan,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    propagation_start: str,
) -> tuple[dict[int, Any], list[dict[str, Any]], int]:
    """Register all anchors in one tracker session, then propagate once."""
    session_id = base.start_session(
        predictor,
        frames,
        sampled_indices,
        offload_video_to_cpu,
        offload_state_to_cpu,
    )
    logs: list[dict[str, Any]] = []
    try:
        predictor.handle_request(request={"type": "reset_session", "session_id": session_id})
        for anchor in sorted(plan.anchors, key=lambda item: item.sam_frame_idx):
            for box in sorted(anchor.boxes, key=lambda item: item.obj_id):
                request = base.build_instance_box_prompt_request(
                    session_id,
                    anchor.sam_frame_idx,
                    box.obj_id,
                    box.box_2d,
                )
                response = predictor.handle_request(request=request)
                prompt_ids = base.output_obj_ids(response.get("outputs"))
                if box.obj_id not in prompt_ids:
                    raise RuntimeError(
                        f"SAM prompt at frame {anchor.frame_idx} omitted obj_id={box.obj_id}; "
                        f"returned {prompt_ids}"
                    )
                logs.append(
                    {
                        "status": "ok",
                        "original_frame_idx": anchor.frame_idx,
                        "sam_frame_idx": anchor.sam_frame_idx,
                        "obj_id": box.obj_id,
                        "box_2d": box.box_2d,
                        "prompt_method": "instance_tracker_native_box",
                        "tracker_box_corner_points": request["points"],
                        "tracker_box_labels": request["point_labels"],
                    }
                )
        ordered = sorted(plan.anchors, key=lambda item: item.sam_frame_idx)
        if propagation_start == "first":
            start_anchor = ordered[0]
        elif propagation_start == "last":
            start_anchor = ordered[-1]
        else:
            # Starting propagation at the terminal sampled frame makes SAM3
            # emit masks ONLY on conditioning frames (observed on video_6185:
            # anchors [0, 510/510] -> 2 non-empty frames, HOTA 0.09; same
            # anchors started at frame 0 -> HOTA 0.81). Terminal anchors are
            # therefore ineligible when an alternative exists; among eligible
            # anchors pick the one closest to the middle of the video.
            last_index = len(sampled_indices) - 1
            candidates = [a for a in ordered if a.sam_frame_idx < last_index] or ordered
            start_anchor = min(
                candidates, key=lambda a: abs(a.sam_frame_idx - last_index / 2)
            )
        raw = base.remap_outputs(
            base.propagate(predictor, session_id, start_anchor.sam_frame_idx),
            sampled_indices,
        )
        return raw, logs, start_anchor.frame_idx
    finally:
        base.close_session(predictor, session_id)


def single_object_plan(
    plan: SavedMultiAnchorPlan, obj_id: int
) -> SavedMultiAnchorPlan:
    """Keep every anchor for one stable object identity.

    This deliberately isolates only the tracker-session topology. Multiple
    anchors for the same object remain conditioning frames in one session;
    splitting those anchors into separate runs would additionally test the
    nearest-anchor stitching policy used by the trained v1/v2 replays.
    """
    targets = [target for target in plan.targets if target.obj_id == obj_id]
    if len(targets) != 1:
        raise ValueError(f"expected one target for obj_id={obj_id}, got {len(targets)}")
    anchors: list[SavedAnchor] = []
    for anchor in plan.anchors:
        boxes = [box for box in anchor.boxes if box.obj_id == obj_id]
        if not boxes:
            continue
        anchors.append(
            anchor.model_copy(update={"obj_ids": [obj_id], "boxes": boxes})
        )
    if not anchors:
        raise ValueError(f"plan has no anchor box for obj_id={obj_id}")
    return plan.model_copy(update={"targets": targets, "anchors": anchors})


def merge_independent_outputs(
    outputs_by_object: dict[int, dict[int, Any]],
) -> dict[int, Any]:
    """Merge single-object session outputs into the normal joint raw format."""
    by_frame: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for expected_obj_id, raw in sorted(outputs_by_object.items()):
        for frame_idx, output in raw.items():
            if not isinstance(output, dict):
                raise TypeError(
                    f"obj_id={expected_obj_id} frame={frame_idx}: output must be a dict"
                )
            obj_ids = np.asarray(output.get("out_obj_ids", [])).reshape(-1)
            if obj_ids.size and set(map(int, obj_ids)) != {int(expected_obj_id)}:
                raise ValueError(
                    f"obj_id={expected_obj_id} frame={frame_idx}: "
                    f"session returned ids={list(map(int, obj_ids))}"
                )
            by_frame.setdefault(int(frame_idx), []).append((expected_obj_id, output))

    merged: dict[int, Any] = {}
    for frame_idx, object_outputs in sorted(by_frame.items()):
        object_outputs.sort(key=lambda item: item[0])
        parts = [output for _, output in object_outputs]
        id_counts = [
            np.asarray(part.get("out_obj_ids", [])).reshape(-1).size
            for part in parts
        ]
        keys = set().union(*(part.keys() for part in parts))
        frame_output: dict[str, Any] = {}
        for key in sorted(keys):
            values = [part.get(key) for part in parts]
            arrays = [np.asarray(value) for value in values]
            object_aligned = all(
                array.ndim >= 1 and array.shape[0] == count
                for array, count in zip(arrays, id_counts)
            )
            if object_aligned:
                frame_output[key] = np.concatenate(arrays, axis=0)
                continue
            non_null = [value for value in values if value is not None]
            if not non_null:
                frame_output[key] = None
                continue
            first = non_null[0]
            same_values = all(
                np.array_equal(value, first)
                if isinstance(value, np.ndarray) or isinstance(first, np.ndarray)
                else value == first
                for value in non_null[1:]
            )
            if not same_values:
                raise ValueError(
                    f"frame={frame_idx}: non-object field {key!r} differs across sessions"
                )
            frame_output[key] = first
        merged[frame_idx] = frame_output
    return merged


def run_per_object_sessions(
    predictor: Any,
    frames: Sequence[Any],
    sampled_indices: Sequence[int],
    plan: SavedMultiAnchorPlan,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    propagation_start: str,
) -> tuple[dict[int, Any], list[dict[str, Any]], dict[int, int]]:
    """Run one multi-anchor SAM3 session per target object, then merge outputs."""
    outputs_by_object: dict[int, dict[int, Any]] = {}
    starts: dict[int, int] = {}
    logs: list[dict[str, Any]] = []
    for target in sorted(plan.targets, key=lambda item: item.obj_id):
        obj_id = int(target.obj_id)
        object_plan = single_object_plan(plan, obj_id)
        raw, object_logs, start_frame = run_multi_anchor_session(
            predictor,
            frames,
            sampled_indices,
            object_plan,
            offload_video_to_cpu,
            offload_state_to_cpu,
            propagation_start,
        )
        outputs_by_object[obj_id] = raw
        starts[obj_id] = int(start_frame)
        logs.extend(
            {**entry, "session_mode": "per-object", "session_obj_id": obj_id}
            for entry in object_logs
        )
    return merge_independent_outputs(outputs_by_object), logs, starts


def timestamp_sampled_frames(
    num_frames: int,
    source_fps: float,
    target_fps: float,
) -> list[int]:
    """Sample source indices by timestamp, including frame zero exactly."""
    if num_frames < 1:
        return []
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    duration = (num_frames - 1) / source_fps
    count = int(math.floor(duration * target_fps)) + 1
    return sorted({
        min(num_frames - 1, int(round(sample_idx * source_fps / target_fps)))
        for sample_idx in range(count)
    })

def segment_boundary_dense_frames(
    num_frames: int,
    stride: int,
    selection: AnchorSelectionResponse,
    radius: int,
    base_indices: Sequence[int] | None = None,
    required_frame_stride: int = 0,
    base_sampling_metadata: dict[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Sampling grid plus anchors and dense presence-boundary windows."""
    boundaries = {selection.first_visible_frame_idx, selection.last_visible_frame_idx}
    num_segments = 0
    for target in selection.targets:
        for segment in target.visibility_segments:
            num_segments += 1
            boundaries.add(segment.first_frame_idx)
            boundaries.add(segment.last_frame_idx)
    boundaries = {min(max(0, b), num_frames - 1) for b in boundaries}
    base_grid = set(
        base.stride_frames(num_frames, stride) if base_indices is None else base_indices
    )
    required_grid = (
        set(range(0, num_frames, required_frame_stride))
        if required_frame_stride > 0 else set()
    )
    anchor_frames = {int(anchor.frame_idx) for anchor in selection.anchors}
    dense: set[int] = set()
    for boundary in boundaries:
        dense.update(
            range(max(0, boundary - radius), min(num_frames - 1, boundary + radius) + 1)
        )
    combined = sorted(base_grid | required_grid | anchor_frames | dense | {num_frames - 1})
    metadata = {
        "sampling_mode": (
            "timestamp_fps_plus_required_grid_anchors_and_boundaries"
            if base_indices is not None else
            "stride_plus_required_grid_anchors_and_boundaries"
        ),
        "sam_stride": stride,
        "required_frame_stride": required_frame_stride,
        "boundary_radius": radius,
        "first_visible_frame_idx": selection.first_visible_frame_idx,
        "last_visible_frame_idx": selection.last_visible_frame_idx,
        "num_visibility_segments": num_segments,
        "segment_boundaries": sorted(boundaries),
        "num_base_stride_frames": len(base_grid),
        "num_base_sampled_frames": len(base_grid),
        "base_source_indices": sorted(base_grid),
        "num_required_grid_frames": len(required_grid),
        "required_source_indices": sorted(required_grid),
        "num_anchor_frames": len(anchor_frames),
        "anchor_source_indices": sorted(anchor_frames),
        "num_additional_frames": len(combined) - len(base_grid),
        "num_sam_frames": len(combined),
    }
    metadata.update(base_sampling_metadata or {})
    return combined, metadata



def apply_segment_gating(
    raw: dict[int, Any],
    selection: AnchorSelectionResponse,
    margin: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Drop each object's outputs outside its visibility segments ± margin.

    SAM3 keeps emitting confident masks for absent lookalike objects (its
    out_probs stay at 1.0), so the Gemini visibility segments are the only
    signal for when an object can actually be on screen.  A target without
    segments (older RL action JSONs) falls back to the global bounds.
    """
    intervals: dict[int, list[tuple[int, int]]] = {}
    for target in selection.targets:
        spans = [
            (segment.first_frame_idx, segment.last_frame_idx)
            for segment in target.visibility_segments
        ] or [(selection.first_visible_frame_idx, selection.last_visible_frame_idx)]
        intervals[target.obj_id] = [(first - margin, last + margin) for first, last in spans]

    dropped_per_obj: dict[int, int] = {obj_id: 0 for obj_id in intervals}
    gated: dict[int, Any] = {}
    for frame_idx, out in raw.items():
        obj_ids = np.atleast_1d(np.asarray(out["out_obj_ids"]))
        keep = np.array(
            [
                any(
                    first <= frame_idx <= last
                    for first, last in intervals.get(int(obj_id), [(frame_idx, frame_idx)])
                )
                for obj_id in obj_ids
            ],
            dtype=bool,
        )
        if keep.all():
            gated[frame_idx] = out
            continue
        for obj_id in obj_ids[~keep]:
            dropped_per_obj[int(obj_id)] = dropped_per_obj.get(int(obj_id), 0) + 1
        filtered = dict(out)
        for key, value in out.items():
            array = np.asarray(value)
            if array.ndim >= 1 and array.shape[0] == len(obj_ids):
                filtered[key] = array[keep]
        gated[frame_idx] = filtered
    return gated, {
        "enabled": True,
        "margin": margin,
        "intervals": {str(obj_id): spans for obj_id, spans in intervals.items()},
        "num_dropped_outputs": int(sum(dropped_per_obj.values())),
        "dropped_per_obj": {str(obj_id): count for obj_id, count in dropped_per_obj.items()},
    }


def compact_dense_tracks(raw: dict[int, Any], num_frames: int) -> list[dict[str, Any]]:
    """Convert v7 SAM outputs directly to dense normalized XYXY tracks.

    This is the established v7 submission conversion, performed while the raw
    masks are still in RAM so blind-test inference never needs a large pickle.
    Missing stride frames are previous-value filled (or next-value filled before
    the first observation), matching the historical v7 submission converter.
    """
    per_object: dict[int, dict[int, list[float]]] = {}
    for frame_idx, output in raw.items():
        obj_ids = np.asarray(output.get("out_obj_ids", [])).reshape(-1)
        boxes = np.asarray(output.get("out_boxes_xywh", []), dtype=np.float64)
        if boxes.size == 0:
            continue
        boxes = boxes.reshape(-1, 4)
        for index, obj_id in enumerate(obj_ids):
            if index >= len(boxes):
                continue
            x, y, width, height = map(float, boxes[index])
            if not (np.isfinite(boxes[index]).all() and width > 0 and height > 0):
                continue
            per_object.setdefault(int(obj_id), {})[int(frame_idx)] = [
                x, y, x + width, y + height,
            ]

    tracks: list[dict[str, Any]] = []
    for track_id, (_, by_frame) in enumerate(sorted(per_object.items())):
        known = sorted(by_frame)
        if not known:
            continue
        dense_boxes: list[list[float]] = []
        known_index = 0
        for frame_idx in range(num_frames):
            while known_index + 1 < len(known) and known[known_index + 1] <= frame_idx:
                known_index += 1
            source_frame = (
                known[known_index] if known[known_index] <= frame_idx else known[0]
            )
            dense_boxes.append(by_frame[source_frame])
        tracks.append(
            {
                "id": track_id,
                "score": 1.0,
                "frame_ids": list(range(num_frames)),
                "bounding_boxes": dense_boxes,
            }
        )
    return tracks


def make_rl_episode(
    allowed_frames: Sequence[int],
    selection: AnchorSelectionResponse,
    policy: str,
    max_anchor_frames: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reward": None,
        "reward_source": "external evaluator only; no GT is sent to Gemini or the policy",
        "action_space": {
            "candidate_anchor_frames": list(map(int, allowed_frames)),
            "max_anchor_frames": max_anchor_frames,
            "action_format": "AnchorSelectionResponse JSON passed with --anchor-selection-json",
        },
        "selected_action": selection.model_dump(mode="json"),
        "selection_policy": policy,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    parser = argparse.ArgumentParser(
        description="pipeline-v7 multi-anchor Gemini bbox -> one SAM3 propagation"
    )
    parser.add_argument("--video-id", "--video_id", dest="video_id", required=True)
    parser.add_argument("--question-id", "--question_id", dest="question_id", required=True)
    parser.add_argument("--sam-stride", "--sam_stride", dest="sam_stride", type=int, default=10)
    parser.add_argument(
        "--sam-target-fps",
        type=float,
        help="sample SAM frames by timestamps at this FPS instead of a fixed integer stride",
    )
    parser.add_argument(
        "--required-frame-stride",
        type=int,
        default=0,
        help="also include every Nth source frame; use 30 to retain the HOTA grid",
    )
    parser.add_argument("--boundary-radius", "--boundary_radius", dest="boundary_radius", type=int, default=9)
    parser.add_argument("--max-anchor-frames", type=int, default=3)
    parser.add_argument(
        "--anchor-selection-json",
        type=Path,
        help="RL/manual AnchorSelectionResponse action; bypasses Gemini anchor selection",
    )
    parser.add_argument(
        "--reuse-plan-json",
        "--reuse_plan_json",
        dest="reuse_plan_json",
        type=Path,
        help="reuse a saved multi_anchor_plan.json (noun phrase, anchors, and boxes); skips every Gemini call",
    )
    parser.add_argument(
        "--verify-noun",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="critic pass: re-derive the answer and override a wrong noun phrase (+1 Gemini call)",
    )
    parser.add_argument(
        "--segment-gate-margin",
        "--segment_gate_margin",
        dest="segment_gate_margin",
        type=int,
        default=15,
        help="drop object outputs outside their visibility segments ± this many frames; negative disables gating",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--propagation-start",
        choices=("first", "middle", "last"),
        default="middle",
        help="which registered anchor starts the single bidirectional propagation",
    )
    parser.add_argument(
        "--session-mode",
        choices=("joint", "per-object"),
        default="joint",
        help=(
            "joint registers every target in one SAM3 session; per-object uses "
            "one session per stable obj_id while retaining all of that object's anchors"
        ),
    )
    parser.add_argument("--vis-stride", type=int, default=60)
    parser.add_argument(
        "--retain-ungated-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain a separate ungated pickle when presence gating changed outputs",
    )
    parser.add_argument(
        "--render-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render periodic tracking galleries",
    )
    parser.add_argument(
        "--compact-box-output",
        action="store_true",
        help=(
            "write per-sample dense submission boxes to predictions.json and "
            "do not write raw/refined/baseline mask pickles"
        ),
    )
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", base.DEFAULT_MODEL))
    parser.add_argument("--gemini-video-max-frames", type=int, default=192)
    parser.add_argument("--gemini-video-max-dimension", type=int, default=512)
    parser.add_argument(
        "--gemini-video-fps",
        type=float,
        default=4.0,
        help="Gemini video sampling rate for the planning proxy (API default is 1fps)",
    )
    parser.add_argument("--google-cloud-project", default=project, required=not project)
    parser.add_argument("--google-cloud-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    predictor: Any | None = None,
    frames_cache: dict[str, tuple[list[Any], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if args.sam_stride < 1:
        raise ValueError("--sam-stride must be at least 1")
    if args.sam_target_fps is not None and args.sam_target_fps <= 0:
        raise ValueError("--sam-target-fps must be positive")
    if args.required_frame_stride < 0:
        raise ValueError("--required-frame-stride must be non-negative")
    if args.boundary_radius < 0:
        raise ValueError("--boundary-radius must be non-negative")
    if args.max_anchor_frames < 1:
        raise ValueError("--max-anchor-frames must be at least 1")

    video_id, question_id = str(args.video_id), str(args.question_id)
    run_dir = Path(args.output_root) / f"{video_id}_q{question_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = base.video_path_for(video_id)
    question: str
    cache_key = str(video_path.resolve())
    if frames_cache is not None and cache_key in frames_cache:
        frames, video_metadata = frames_cache[cache_key]
    else:
        frames, video_metadata = base.load_video_frames(video_path)
        if frames_cache is not None:
            # Keep only the current video; test questions are grouped by video.
            frames_cache.clear()
            frames_cache[cache_key] = (frames, video_metadata)
    if args.sam_target_fps is None:
        base_sampled_indices = base.stride_frames(len(frames), args.sam_stride)
        base_sampling_metadata = {
            "source_fps": video_metadata["fps"],
            "sam_target_fps": None,
        }
    else:
        base_sampled_indices = timestamp_sampled_frames(
            len(frames), float(video_metadata["fps"]), args.sam_target_fps
        )
        base_sampling_metadata = {
            "source_fps": video_metadata["fps"],
            "sam_target_fps": args.sam_target_fps,
            "timestamp_formula": "round(sample_idx * source_fps / sam_target_fps)",
        }
    allowed_frames = base_sampled_indices
    reused_plan: SavedMultiAnchorPlan | None = None
    if args.reuse_plan_json:
        reused_plan = SavedMultiAnchorPlan.model_validate_json(
            args.reuse_plan_json.read_text(encoding="utf-8")
        )
        if (reused_plan.video_id, reused_plan.question_id) != (video_id, question_id):
            raise ValueError(
                f"--reuse-plan-json is for {reused_plan.video_id}_q{reused_plan.question_id}, "
                f"not {video_id}_q{question_id}"
            )
        question = reused_plan.question
        proxy_metadata = {"reused_plan_json": str(args.reuse_plan_json.resolve())}
        noun_phrase = reused_plan.answer_noun_phrase
        noun_reasoning = reused_plan.noun_reasoning
        noun_confidence = reused_plan.noun_confidence
        noun_metadata = {"source": "reused_plan_json"}
        selection = AnchorSelectionResponse(
            decision_summary=reused_plan.decision_summary,
            first_visible_frame_idx=reused_plan.first_visible_frame_idx,
            last_visible_frame_idx=reused_plan.last_visible_frame_idx,
            visibility_boundary_reason=reused_plan.visibility_boundary_reason,
            targets=reused_plan.targets,
            anchors=[
                AnchorSelection(
                    frame_idx=anchor.frame_idx,
                    obj_ids=anchor.obj_ids,
                    reason=anchor.reason,
                )
                for anchor in reused_plan.anchors
            ],
        )
        selection_metadata = {"source": "reused_plan_json"}
        anchor_policy = "reused_plan_json"
        gemini_model = reused_plan.model
    else:
        question = base.load_question(video_id, question_id)
        proxy_path, proxy_metadata = base.build_planning_proxy(
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
        noun, noun_metadata = request_noun_phrase(
            client, args.gemini_model, proxy_path, question, args.gemini_video_fps
        )
        noun_phrase = noun.answer_noun_phrase
        noun_reasoning = noun.reasoning
        noun_confidence = noun.confidence
        gemini_model = args.gemini_model
        instance_count_hint = None
        if args.verify_noun:
            verification, verification_metadata = verify_noun_phrase(
                client,
                args.gemini_model,
                proxy_path,
                question,
                noun_phrase,
                args.gemini_video_fps,
            )
            corrected = (verification.get("corrected_noun_phrase") or "").strip()
            if verification.get("verdict") == "wrong" and corrected:
                noun_reasoning = (
                    f"[critic override of '{noun_phrase}'] "
                    + str(verification.get("reasoning", ""))
                )
                noun_phrase = corrected
            count = verification.get("expected_instance_count")
            if isinstance(count, int) and count > 1:
                instance_count_hint = count
            noun_metadata = {
                **noun_metadata,
                "noun_verification": {**verification, "call_metadata": verification_metadata},
            }
        if args.anchor_selection_json:
            selection, selection_metadata = load_anchor_selection(
                args.anchor_selection_json,
                allowed_frames,
                args.max_anchor_frames,
            )
            anchor_policy = "external_json"
        else:
            selection, selection_metadata = request_anchor_selection(
                client,
                args.gemini_model,
                proxy_path,
                question,
                noun_phrase,
                allowed_frames,
                args.max_anchor_frames,
                args.gemini_video_fps,
                instance_count_hint=instance_count_hint,
            )
            anchor_policy = "gemini"
    selection_path = base.save_json(
        selection.model_dump(mode="json"), run_dir / "anchor_selection.json"
    )
    base.save_json(
        make_rl_episode(allowed_frames, selection, anchor_policy, args.max_anchor_frames),
        run_dir / "rl_episode.json",
    )
    if args.plan_only:
        return {
            "status": "plan_only",
            "anchor_selection": str(selection_path),
            "rl_episode": str(run_dir / "rl_episode.json"),
        }

    sampled_indices, sampling = segment_boundary_dense_frames(
        len(frames),
        args.sam_stride,
        selection,
        args.boundary_radius,
        base_indices=base_sampled_indices if args.sam_target_fps is not None else None,
        required_frame_stride=args.required_frame_stride,
        base_sampling_metadata=base_sampling_metadata,
    )
    reused_anchors_by_frame = (
        {anchor.frame_idx: anchor for anchor in reused_plan.anchors}
        if reused_plan is not None
        else {}
    )
    identity_by_id = {target.obj_id: target for target in selection.targets}
    reused_boxes_by_frame = (
        {anchor.frame_idx: anchor.boxes for anchor in reused_plan.anchors}
        if reused_plan is not None
        else {}
    )
    saved_anchors: list[SavedAnchor] = []
    for anchor in selection.anchors:
        if args.compact_box_output and reused_plan is not None:
            exact_path = Path(reused_anchors_by_frame[anchor.frame_idx].exact_frame_path)
        else:
            exact_path = base.save_rgb(
                frames[anchor.frame_idx],
                run_dir / "anchor_frames" / f"frame_{anchor.frame_idx:05d}.png",
            )
        if reused_plan is not None:
            anchor_boxes = reused_boxes_by_frame[anchor.frame_idx]
            metadata = {"source": "reused_plan_json"}
        else:
            boxes, metadata = request_anchor_boxes(
                client,
                args.gemini_model,
                proxy_path,
                exact_path,
                question,
                noun_phrase,
                anchor,
                [identity_by_id[obj_id] for obj_id in anchor.obj_ids],
                args.gemini_video_fps,
            )
            anchor_boxes = boxes.boxes
        saved_anchors.append(
            SavedAnchor(
                frame_idx=anchor.frame_idx,
                sam_frame_idx=sampled_indices.index(anchor.frame_idx),
                obj_ids=anchor.obj_ids,
                reason=anchor.reason,
                boxes=anchor_boxes,
                exact_frame_path=str(exact_path),
                call_metadata=metadata,
            )
        )
    plan = SavedMultiAnchorPlan(
        video_id=video_id,
        question_id=question_id,
        question=question,
        answer_noun_phrase=noun_phrase,
        noun_reasoning=noun_reasoning,
        noun_confidence=noun_confidence,
        decision_summary=selection.decision_summary,
        first_visible_frame_idx=selection.first_visible_frame_idx,
        last_visible_frame_idx=selection.last_visible_frame_idx,
        visibility_boundary_reason=selection.visibility_boundary_reason,
        targets=selection.targets,
        anchors=saved_anchors,
        sam_stride=args.sam_stride,
        sam_target_fps=args.sam_target_fps,
        required_frame_stride=args.required_frame_stride or None,
        boundary_radius=args.boundary_radius,
        max_anchor_frames=args.max_anchor_frames,
        anchor_policy=anchor_policy,
        model=gemini_model,
        gemini_video_fps=args.gemini_video_fps,
        call_metadata={
            "noun_phrase": noun_metadata,
            "anchor_selection": selection_metadata,
            "gemini_planning_proxy": proxy_metadata,
        },
    )
    base.save_json(plan.model_dump(mode="json"), run_dir / "multi_anchor_plan.json")
    base.save_json(
        {
            **sampling,
            "num_original_frames": len(frames),
            "source_fps": video_metadata["fps"],
            "sam_target_fps": args.sam_target_fps,
            "required_frame_stride": args.required_frame_stride,
            "anchors": [anchor.model_dump(mode="json") for anchor in plan.anchors],
            "sam_to_original_frame": {
                str(index): frame_idx for index, frame_idx in enumerate(sampled_indices)
            },
        },
        run_dir / "sam_frame_map.json",
    )
    if not args.compact_box_output:
        draw_anchor_plan(frames, plan, run_dir / "anchor_overlays")

    if not base.torch.cuda.is_available():
        raise RuntimeError("SAM3 video inference requires CUDA")
    if predictor is None:
        predictor = base.build_sam3_video_predictor(gpus_to_use=[base.torch.cuda.current_device()])
    if args.session_mode == "joint":
        raw, prompt_log, joint_start = run_multi_anchor_session(
            predictor,
            frames,
            sampled_indices,
            plan,
            args.offload_video_to_cpu,
            args.offload_state_to_cpu,
            args.propagation_start,
        )
        propagation_start_frames = {"joint": int(joint_start)}
    else:
        raw, prompt_log, object_starts = run_per_object_sessions(
            predictor,
            frames,
            sampled_indices,
            plan,
            args.offload_video_to_cpu,
            args.offload_state_to_cpu,
            args.propagation_start,
        )
        propagation_start_frames = {
            str(obj_id): int(frame_idx)
            for obj_id, frame_idx in object_starts.items()
        }
    base.save_json(prompt_log, run_dir / "multi_anchor_prompt_execution_log.json")
    ungated = raw
    if args.segment_gate_margin >= 0:
        raw, gating = apply_segment_gating(raw, selection, args.segment_gate_margin)
    else:
        gating = {"enabled": False}
    base.save_json(gating, run_dir / "segment_gating.json")
    if args.compact_box_output:
        tracks = compact_dense_tracks(raw, len(frames))
        compact_prediction = {
            video_id: {"grounded_question": {question_id: tracks}}
        }
        compact_path = base.save_json(compact_prediction, run_dir / "predictions.json")
        for name in (
            "raw_outputs_true.pkl",
            "refined_raw_outputs_true.pkl",
            "baseline_raw_outputs_true.pkl",
        ):
            (run_dir / name).unlink(missing_ok=True)
        raw_path = None
    else:
        # The three pickles are hardlinked unless their content actually
        # differs; a full copy of the mask dict is ~0.5-1 GB per run.
        raw_path = base.save_pickle(raw, run_dir / "raw_outputs_true.pkl")
        for name in ("refined_raw_outputs_true.pkl", "baseline_raw_outputs_true.pkl"):
            (run_dir / name).unlink(missing_ok=True)
        os.link(raw_path, run_dir / "refined_raw_outputs_true.pkl")
        if gating.get("num_dropped_outputs") and args.retain_ungated_baseline:
            base.save_pickle(ungated, run_dir / "baseline_raw_outputs_true.pkl")
        else:
            os.link(raw_path, run_dir / "baseline_raw_outputs_true.pkl")
        compact_path = None
    summary = base.summarize(raw, len(frames))
    summary.update(
        {
            "video_id": video_id,
            "question_id": question_id,
            "question": question,
            "pipeline_variant": "pipeline-v7-multi-anchor-box-baseline",
            "initialization": "multi_anchor_instance_tracker_boxes",
            "session_mode": args.session_mode,
            "num_tracker_sessions": (
                1 if args.session_mode == "joint" else len(plan.targets)
            ),
            "sam_stride": args.sam_stride,
            "anchor_policy": anchor_policy,
            "anchor_frames": [anchor.frame_idx for anchor in plan.anchors],
            "propagation_start_frame": (
                next(iter(propagation_start_frames.values()))
                if len(propagation_start_frames) == 1
                else None
            ),
            "propagation_start_frames": propagation_start_frames,
            "num_input_boxes": sum(len(anchor.boxes) for anchor in plan.anchors),
            "num_sam_frames": len(sampled_indices),
            "segment_gating": gating,
            "raw_output_path": str(raw_path) if raw_path is not None else None,
            "compact_prediction_path": (
                str(compact_path) if compact_path is not None else None
            ),
        }
    )
    if args.render_visualizations:
        summary["visualization_paths"] = base.render_visualizations(
            frames,
            raw,
            run_dir / "visualizations",
            args.vis_stride,
            [anchor.frame_idx for anchor in plan.anchors],
        )
    else:
        summary["visualization_paths"] = []
    base.save_json(summary, run_dir / "run_summary.json")
    result = {"status": "complete", **summary}
    base.save_json(result, run_dir / "run_result.json")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
