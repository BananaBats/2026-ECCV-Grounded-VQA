#!/usr/bin/env python3
"""Turn GVQA-format predictions into one VidSTG tube per sentence.

Input is any merged prediction file in the pipeline's submission format
({clip_id: {"grounded_question": {qid: [track, ...]}}}, normalized xyxy, clip
frame ids): dense replay, sparse fusion, or the finalized fusion output.

For each sentence the exporter
  1. keeps a single track (obj_id 0, else the longest),
  2. crops it to the predicted temporal segment from the saved plan,
  3. converts boxes to pixel xyxy and frame ids to ORIGINAL video frame ids.

--temporal-source:
  grounding  plan grounding_segment (default; the VidSTG setting)
  visibility plan first/last_visible_frame_idx (ablation)
  grounding_sparse
             plan grounding_segment, trimmed to the sparse Gemini frames where
             the target was seen (± half the sparse stride); never widened.
             Needs --sparse-root.  Falls back to grounding without sparse.
  full       no temporal cropping (ablation)
GT temporal oracle cropping is an evaluator option (vidstg_eval.py
--oracle-temporal) so this file never reads ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def load_subset(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            video_id, question_id = line.split("\t")
            rows.append((video_id.strip(), question_id.strip()))
    return rows


def choose_track(tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tracks:
        return None
    for track in tracks:
        if int(track.get("id", -1)) == 0:
            return track
    return max(tracks, key=lambda track: len(track.get("frame_ids", [])))


def sparse_seen_frames(sample_dir: Path, object_id: int) -> tuple[list[int], int] | None:
    """Queried sparse frames where the target got a box, and the sparse stride."""
    manifest_path = sample_dir / "manifest.json"
    predictions_path = sample_dir / "sparse_predictions.json"
    if not manifest_path.is_file() or not predictions_path.is_file():
        return None
    manifest = load_json(manifest_path)
    queried = {int(frame) for frame in manifest.get("frame_indices", [])}
    by_frame = load_json(predictions_path).get("predictions_by_frame", {})
    seen = sorted(
        int(frame) for frame, objects in by_frame.items()
        if int(frame) in queried
        and any(int(item["object_id"]) == object_id for item in objects)
    )
    return seen, int(manifest.get("frame_stride", 30))


def trim_to_sparse(
    first: int, last: int, sparse: tuple[list[int], int] | None
) -> tuple[int, int]:
    """Shrink [first, last] to where sparse saw the target; never widen it."""
    if sparse is None or not sparse[0]:
        return first, last
    seen, stride = sparse
    margin = stride // 2
    new_first = max(first, seen[0] - margin)
    new_last = min(last, seen[-1] + margin)
    if new_first > new_last:
        return first, last
    return new_first, new_last


def predicted_segment(
    plan: dict[str, Any] | None,
    source: str,
    num_frames: int,
    sparse: tuple[list[int], int] | None = None,
) -> tuple[int, int]:
    """Half-open clip-frame segment [first, last + 1)."""
    if source == "full" or plan is None:
        return 0, num_frames
    if source in ("grounding", "grounding_sparse"):
        segment = plan.get("grounding_segment")
        if not segment:
            raise ValueError("plan has no grounding_segment (re-plan with --task vidstg)")
        first, last = int(segment["first_frame_idx"]), int(segment["last_frame_idx"])
        if source == "grounding_sparse":
            first, last = trim_to_sparse(first, last, sparse)
    elif source == "visibility":
        first = int(plan["first_visible_frame_idx"])
        last = int(plan["last_visible_frame_idx"])
    else:
        raise ValueError(f"unknown temporal source {source!r}")
    first = min(max(0, first), num_frames - 1)
    last = min(max(first, last), num_frames - 1)
    return first, last + 1


def export_sentence(
    tracks: list[dict[str, Any]] | None,
    plan: dict[str, Any] | None,
    clip: dict[str, Any],
    source: str,
    sparse_dir: Path | None = None,
) -> dict[str, Any]:
    offset = int(clip["begin_fid"])
    num_frames = int(clip["num_frames"])
    width, height = float(clip["width"]), float(clip["height"])
    if source != "full" and plan is None:
        # A missing plan is a failed sample; emit an empty tube.
        return {"pred_segment": None, "boxes_xyxy": {}, "status": "missing_plan"}
    track = choose_track(tracks or [])
    sparse = None
    if source == "grounding_sparse" and sparse_dir is not None:
        sparse = sparse_seen_frames(sparse_dir, 0 if track is None else int(track.get("id", 0)))
    begin, end = predicted_segment(plan, source, num_frames, sparse)
    boxes: dict[str, list[float]] = {}
    if track is not None:
        for frame, box in zip(track["frame_ids"], track["bounding_boxes"]):
            frame = int(frame)
            if not begin <= frame < end:
                continue
            x0, y0, x1, y1 = (float(value) for value in box)
            boxes[str(frame + offset)] = [x0 * width, y0 * height, x1 * width, y1 * height]
    return {
        "pred_segment": [begin + offset, end + offset],
        "boxes_xyxy": boxes,
        "track_id": None if track is None else int(track.get("id", 0)),
        "status": "ok" if track is not None else "missing_track",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True, help="dataset/<split>/vidstg_clips.json")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--temporal-source",
                        choices=("grounding", "grounding_sparse", "visibility", "full"),
                        default="grounding")
    parser.add_argument("--sparse-root", type=Path,
                        help="sparse output root (needed for grounding_sparse)")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions = load_json(args.predictions)
    clips = load_json(args.clips)
    output: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for clip_id, question_id in load_subset(args.subset):
        sample = f"{clip_id}_q{question_id}"
        plan_path = args.plan_root / sample / "multi_anchor_plan.json"
        plan = load_json(plan_path) if plan_path.is_file() else None
        tracks = predictions.get(clip_id, {}).get("grounded_question", {}).get(question_id)
        sparse_dir = args.sparse_root / sample if args.sparse_root else None
        record = export_sentence(tracks, plan, clips[clip_id], args.temporal_source, sparse_dir)
        record.update({"clip_id": clip_id, "question_id": question_id})
        output[sample] = record
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    save_json(args.output, {"temporal_source": args.temporal_source, "samples": output})
    print(f"wrote {args.output}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
