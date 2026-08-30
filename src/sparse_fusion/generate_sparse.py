#!/usr/bin/env python3
"""Generate decoder-exact 30-frame Gemini amodal boxes from saved 3.7 plans.

This is Stage 2 only. It never re-answers the question and never edits a saved
multi_anchor_plan.json. Exact source frames 0,30,60,... are extracted with
ffmpeg and sent in bounded image batches.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparse_common import (
    DEFAULT_PLAN_ROOT,
    DEFAULT_SPARSE_ROOT,
    DEFAULT_SUBSET,
    DEFAULT_VIDEO_ROOT,
    compact_plan,
    expected_ids,
    load_json,
    load_subset,
    sample_name,
    sane_yxyx,
    save_json,
    select_rows,
    yxyx_1000_to_xyxy_normalized,
)


FRAME_STRIDE = 30


class SparseObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: int = Field(ge=0, le=31)
    visibility: Literal["visible", "partially_occluded", "fully_occluded"]
    box_2d: list[int] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("box_2d")
    @classmethod
    def valid_box(cls, value: list[int]) -> list[int]:
        values = [int(item) for item in value]
        if any(item < 0 or item > 1000 for item in values):
            raise ValueError("box_2d values must be in 0..1000")
        y0, x0, y1, x1 = values
        if y1 <= y0 or x1 <= x0:
            raise ValueError("box_2d must have positive area")
        return values


class SparseFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    objects: list[SparseObject] = Field(max_length=8)


class SparseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames: list[SparseFrame] = Field(min_length=1)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def probe_video(video_path: Path) -> dict[str, Any]:
    result = run_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of", "json", str(video_path),
    ])
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    numerator, denominator = (float(value) for value in rate.split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator if denominator else 0.0,
        "metadata_frame_count": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
    }


def extract_sparse_frames(video_path: Path, frame_dir: Path) -> tuple[list[int], dict[str, Any]]:
    """Decode once and cache exact not(mod(n,30)) JPEGs by source frame ID."""
    manifest_path = frame_dir / "frame_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        indices = [int(value) for value in manifest["frame_indices"]]
        if indices and all((frame_dir / f"frame_{idx:06d}.jpg").is_file() for idx in indices):
            return indices, manifest["video"]

    video = probe_video(video_path)
    frame_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sparse_frames_", dir=frame_dir.parent) as tmp_name:
        tmp_dir = Path(tmp_name)
        run_command([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-vf", f"select=not(mod(n\\,{FRAME_STRIDE}))",
            "-vsync", "vfr", "-q:v", "2", "-start_number", "0",
            str(tmp_dir / "selected_%06d.jpg"),
        ])
        selected = sorted(tmp_dir.glob("selected_*.jpg"))
        if not selected:
            raise RuntimeError(f"ffmpeg extracted no frames from {video_path}")
        indices = [index * FRAME_STRIDE for index in range(len(selected))]
        for source, frame_idx in zip(selected, indices):
            os.replace(source, frame_dir / f"frame_{frame_idx:06d}.jpg")

    save_json(manifest_path, {
        "video_path": str(video_path),
        "frame_stride": FRAME_STRIDE,
        "frame_indices": indices,
        "video": video,
    })
    return indices, video


STAGE2_V2_RULES = """
Visibility calibration: use partially_occluded only when a substantial part of
an object is hidden behind another scene object. An object merely held or
touched by a hand is visible. For partially_occluded objects keep the box tight
to the object's actual physical extent; extend it only over the hidden part you
can confidently infer, and never inflate it beyond the object's true size.
""".strip()


def build_prompt(plan: dict[str, Any], frame_indices: list[int], video: dict[str, Any]) -> str:
    plan_json = json.dumps(compact_plan(plan), ensure_ascii=False)
    visible_required = {
        str(frame): sorted(expected_ids(plan, frame)) for frame in frame_indices
    }
    known_ids = sorted(int(target["obj_id"]) for target in plan.get("targets", []))
    indices = ", ".join(str(item) for item in frame_indices)
    return f"""
You are Stage 2. The attached images are decoder-exact source frames in the listed
order: {indices}. Use the following immutable saved target plan; retain its IDs and
never create a new ID:
{plan_json}

For every frame and every target, first identify that target's own visual evidence,
expected physical size, and position from its before/after motion. Then return exactly
one frames record per listed frame_idx and exactly one amodal box_2d
[ymin, xmin, ymax, xmax] normalized 0..1000 plus visibility for every target
physically in-frame.

The only known target IDs are {known_ids}. The saved visibility segments mean
"visibly observed", NOT "physically present". IDs visibly confirmed at each requested
frame are {json.dumps(visible_required)} and must be returned there. A gap between
visible segments is a high-priority occlusion interval: if the same physical object
remains in the scene, return it as partially_occluded or fully_occluded. If it truly
left the scene, omit it. Never use visibility-segment membership to discard a hidden
object. An omitted object means physically absent, not merely invisible.

HARD PROHIBITIONS:
- A box must cover exactly one target identity. Never return a union, pile, word,
  collection, group, shared occlusion region, or any other multi-object box.
- Never copy the same box to two different object_id values. If distinct targets overlap
  or are both occluded, infer a separate full amodal box for each physical instance.
- An occluder is not the target: never return the occluder's whole box, its visible
  fragment, or a target-plus-occluder box. For example, if a small target is hidden
  under a cup, return the inferred small target box, not the cup box.

Use partially_occluded only when some pixels of that specific target are visible, and
box its complete inferred extent rather than the visible fragment. Use fully_occluded
only when zero pixels of that specific target are visible and temporal evidence supports
that it remains in-frame; infer its own location, scale, and full extent from context.
Do not use a large occluder or group box as a substitute for this inference.

Before returning JSON, audit every frame: each object_id must have a distinct physical
instance; each box must match that instance's scale; no box may include another target
or its occluder; and no two distinct IDs may share identical coordinates. The source
resolution is {video['width']}x{video['height']}. These are fixed 30-frame-grid
positions.

{STAGE2_V2_RULES}
""".strip()


def build_contents(types: Any, prompt: str, frame_indices: list[int], frame_dir: Path) -> list[Any]:
    parts = [types.Part.from_text(text=prompt)]
    parts.extend(
        types.Part.from_bytes(
            data=(frame_dir / f"frame_{frame_idx:06d}.jpg").read_bytes(),
            mime_type="image/jpeg",
        )
        for frame_idx in frame_indices
    )
    return [types.Content(role="user", parts=parts)]


def validate_response(
    response: SparseResponse,
    plan: dict[str, Any],
    requested: list[int],
) -> list[SparseObject]:
    returned = [frame.frame_idx for frame in response.frames]
    if returned != requested:
        raise ValueError(f"frame sequence mismatch: expected={requested}, returned={returned}")
    known = {int(target["obj_id"]) for target in plan.get("targets", [])}
    flat: list[SparseObject] = []
    for frame in response.frames:
        ids = [item.object_id for item in frame.objects]
        required = expected_ids(plan, frame.frame_idx)
        if len(ids) != len(set(ids)):
            raise ValueError(f"frame {frame.frame_idx} has duplicate object IDs: {ids}")
        if not set(ids) <= known:
            raise ValueError(
                f"frame {frame.frame_idx} has unknown IDs: returned={ids}, known={sorted(known)}"
            )
        if not required <= set(ids):
            raise ValueError(
                f"frame {frame.frame_idx} misses visibly confirmed IDs: "
                f"required={sorted(required)}, returned={ids}"
            )
        flat.extend(frame.objects)
    return flat


def call_batch(client: Any, types: Any, args: argparse.Namespace, plan: dict[str, Any],
               frame_indices: list[int], frame_dir: Path, video: dict[str, Any],
               call_index: int, call_dir: Path) -> SparseResponse:
    prompt = build_prompt(plan, frame_indices, video)
    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / f"call_{call_index:03d}_prompt.txt").write_text(
        prompt + "\n", encoding="utf-8"
    )
    contents = build_contents(types, prompt, frame_indices, frame_dir)
    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level=args.thinking_level),
        max_output_tokens=args.max_output_tokens,
        response_mime_type="application/json",
        response_schema=SparseResponse,
    )
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, args.max_attempts + 1):
        try:
            print(f"  Gemini call={call_index} frames={frame_indices} attempt={attempt}", flush=True)
            response = client.models.generate_content(model=args.model, contents=contents, config=config)
            parsed = (
                SparseResponse.model_validate(response.parsed)
                if response.parsed is not None
                else SparseResponse.model_validate_json(response.text)
            )
            validate_response(parsed, plan, frame_indices)
            usage = getattr(response, "usage_metadata", None)
            record = {
                "status": "ok", "attempt": attempt, "frame_indices": frame_indices,
                "model": args.model, "model_version": getattr(response, "model_version", None),
                "response_id": getattr(response, "response_id", None),
                "usage_metadata": usage.model_dump(mode="json") if usage else None,
                "response": parsed.model_dump(mode="json"),
            }
            attempts.append(record)
            save_json(call_dir / f"call_{call_index:03d}.json", {"attempts": attempts})
            return parsed
        except Exception as exc:
            raw_text = getattr(locals().get("response"), "text", None)
            candidates = getattr(locals().get("response"), "candidates", None)
            finish_reasons = []
            if candidates:
                finish_reasons = [str(getattr(candidate, "finish_reason", None)) for candidate in candidates]
            usage = getattr(locals().get("response"), "usage_metadata", None)
            attempts.append({
                "status": "error", "attempt": attempt, "frame_indices": frame_indices,
                "error_type": type(exc).__name__, "error": str(exc),
                "finish_reasons": finish_reasons,
                "usage_metadata": usage.model_dump(mode="json") if usage else None,
                "raw_response_text": raw_text,
            })
            save_json(call_dir / f"call_{call_index:03d}.json", {"attempts": attempts})
            message = str(exc).lower()
            retryable = (
                isinstance(exc, ValueError)
                or any(token in message for token in ("429", "500", "502", "503", "504", "timeout"))
            )
            if attempt == args.max_attempts or not retryable:
                raise
            time.sleep(min(10 * (2 ** (attempt - 1)), 40))
    raise RuntimeError("unreachable")


def is_complete(sample_dir: Path) -> bool:
    result = sample_dir / "run_result.json"
    predictions = sample_dir / "sparse_predictions.json"
    return result.is_file() and predictions.is_file() and load_json(result).get("status") == "complete"


def run_sample(client: Any, types: Any, args: argparse.Namespace,
               video_id: str, question_id: str) -> dict[str, Any]:
    sample = sample_name(video_id, question_id)
    sample_dir = args.output_root / sample
    if is_complete(sample_dir) and not args.overwrite:
        print(f"skip {sample} (complete)", flush=True)
        return {"sample": sample, "status": "skipped_complete"}

    plan_path = args.plan_root / sample / "multi_anchor_plan.json"
    video_path = args.video_root / f"{video_id}.mp4"
    if not plan_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(f"missing input for {sample}: plan={plan_path}, video={video_path}")
    plan = load_json(plan_path)
    frame_dir = sample_dir / "frames"
    frame_indices, video = extract_sparse_frames(video_path, frame_dir)
    save_json(sample_dir / "manifest.json", {
        "sample": sample, "video_id": video_id, "question_id": question_id,
        "model": args.model, "frame_stride": FRAME_STRIDE,
        "frames_per_call": args.frames_per_call,
        "thinking_level": args.thinking_level,
        "prompt_contract": "inhong_stage2_v2+saved_visibility_is_not_presence",
        "frame_indices": frame_indices, "plan_path": str(plan_path),
        "video_path": str(video_path), "video": video,
    })
    if args.prepare_only:
        call_dir = sample_dir / "calls"
        call_dir.mkdir(parents=True, exist_ok=True)
        for call_index, start in enumerate(range(0, len(frame_indices), args.frames_per_call)):
            batch = frame_indices[start:start + args.frames_per_call]
            prompt = build_prompt(plan, batch, video)
            (call_dir / f"call_{call_index:03d}_prompt.txt").write_text(
                prompt + "\n", encoding="utf-8"
            )
        print(f"prepared {sample}: {len(frame_indices)} frames", flush=True)
        return {"sample": sample, "status": "prepared", "num_frames": len(frame_indices)}

    all_frames: list[SparseFrame] = []
    call_dir = sample_dir / "calls"
    for call_index, start in enumerate(range(0, len(frame_indices), args.frames_per_call)):
        batch = frame_indices[start:start + args.frames_per_call]
        response = call_batch(client, types, args, plan, batch, frame_dir, video, call_index, call_dir)
        all_frames.extend(response.frames)

    predictions: dict[str, list[dict[str, Any]]] = {str(frame): [] for frame in frame_indices}
    visibility_counts = {"visible": 0, "partially_occluded": 0, "fully_occluded": 0}
    for frame in all_frames:
        for item in frame.objects:
            visibility_counts[item.visibility] += 1
            predictions[str(frame.frame_idx)].append({
                "object_id": item.object_id,
                "visibility": item.visibility,
                "confidence": item.confidence,
                "box_2d_yxyx_1000": item.box_2d,
                "xyxy_normalized": yxyx_1000_to_xyxy_normalized(item.box_2d),
            })
    save_json(sample_dir / "sparse_predictions.json", {
        "video_id": video_id, "question_id": question_id,
        "answer_noun_phrase": plan.get("answer_noun_phrase", ""),
        "frame_stride": FRAME_STRIDE,
        "targets": compact_plan(plan)["targets"],
        "predictions_by_frame": predictions,
    })
    result = {
        "status": "complete", "sample": sample, "model": args.model,
        "num_source_frames": video.get("metadata_frame_count"),
        "num_sparse_frames": len(frame_indices), "num_calls": (len(frame_indices) + args.frames_per_call - 1) // args.frames_per_call,
        "num_boxes": sum(len(items) for items in predictions.values()),
        "visibility_counts": visibility_counts,
    }
    save_json(sample_dir / "run_result.json", result)
    print(f"complete {sample}: frames={len(frame_indices)} boxes={result['num_boxes']}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SPARSE_ROOT)
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--google-cloud-project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--google-cloud-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--frames-per-call", type=int, default=8)
    parser.add_argument("--thinking-level", choices=("minimal", "low", "medium", "high"), default="high")
    # Thinking tokens count against this budget. High thinking plus eight images can
    # consume most of 8K before the structured JSON is complete.
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", action="append", help="only run video_N_qM (repeatable)")
    parser.add_argument("--prepare-only", action="store_true", help="extract exact frames; make no API calls")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames_per_call <= 15:
        parser.error("--frames-per-call must be in 1..15")
    args.plan_root = args.plan_root.resolve()
    args.video_root = args.video_root.resolve()
    args.output_root = args.output_root.resolve()
    return args


def main() -> int:
    args = parse_args()
    rows = select_rows(
        load_subset(args.subset), samples=set(args.sample) if args.sample else None,
        num_shards=args.num_shards, shard_index=args.shard_index, limit=args.limit,
    )
    if not rows:
        raise ValueError("no samples selected")
    client = types = None
    if not args.prepare_only:
        if not args.google_cloud_project:
            raise ValueError(
                "set GOOGLE_CLOUD_PROJECT or pass --google-cloud-project before an external Gemini call"
            )
        from google import genai
        from google.genai import types as genai_types
        types = genai_types
        client = genai.Client(
            vertexai=True, project=args.google_cloud_project,
            location=args.google_cloud_location,
            http_options=types.HttpOptions(api_version="v1", timeout=300_000),
        )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for video_id, question_id in rows:
        try:
            results.append(run_sample(client, types, args, video_id, question_id))
        except Exception as exc:
            sample = sample_name(video_id, question_id)
            failure = {"sample": sample, "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(failure)
            save_json(args.output_root / sample / "run_result.json", {"status": "failed", **failure})
            print(f"FAILED {sample}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    save_json(args.output_root / f"shard_{args.shard_index:03d}_summary.json", {
        "num_selected": len(rows), "num_succeeded": len(results),
        "num_failed": len(failures), "results": results, "failures": failures,
    })
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
