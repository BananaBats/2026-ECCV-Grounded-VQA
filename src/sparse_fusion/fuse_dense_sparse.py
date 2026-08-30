#!/usr/bin/env python3
"""Fuse 30-frame Gemini amodal boxes with pipeline-v7 stride-3 SAM tracks.

The per-target gate mirrors inhong/v13:
  max plan-anchor/dense IoU >= 0.25
  at least 3 shared sparse frames
  mean sparse/dense IoU >= 0.30

Trusted sampled SAM tracks own only the frames they actually cover, while the
Gemini grid owns occlusion gaps. Distrusted targets fall back to the sparse grid.
Missing frames are interpolated only under v13's short-gap/stable-bracket rule.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

from sparse_common import (
    DEFAULT_DENSE_ROOT,
    DEFAULT_FUSED_ROOT,
    DEFAULT_PLAN_ROOT,
    DEFAULT_SPARSE_ROOT,
    DEFAULT_SUBSET,
    box_iou,
    load_json,
    load_subset,
    plan_target_ids,
    sample_name,
    sane_yxyx,
    save_json,
    select_rows,
    xyxy_normalized_to_yxyx_1000,
    yxyx_1000_to_xyxy_normalized,
)


def load_dense_tracks(path: Path, video_id: str, question_id: str) -> dict[int, dict[int, list[float]]]:
    payload = load_json(path)
    tracks = payload[video_id]["grounded_question"][str(question_id)]
    result: dict[int, dict[int, list[float]]] = {}
    for track in tracks:
        frame_ids = [int(value) for value in track["frame_ids"]]
        boxes = track["bounding_boxes"]
        if len(frame_ids) != len(boxes):
            raise ValueError(f"frame/box length mismatch in {path}, track {track.get('id')}")
        object_id = int(track["id"])
        if object_id in result:
            raise ValueError(f"duplicate dense track id {object_id} in {path}")
        result[object_id] = {
            frame_idx: xyxy_normalized_to_yxyx_1000(box)
            for frame_idx, box in zip(frame_ids, boxes)
        }
    return result


def recover_sampled_sam_tracks(
    dense_dir: Path,
    dense_tracks: dict[int, dict[int, list[float]]],
) -> tuple[dict[int, dict[int, list[float]]], dict[str, Any]]:
    """Undo compact export's full-frame hold using pipeline-v7 metadata.

    Exact target-specific SAM failures cannot be recovered because raw masks were
    discarded. The best available reconstruction is the global non-empty output
    schedule intersected with each target's segment-gating intervals.
    """
    result_path = dense_dir / "run_result.json"
    if not result_path.is_file():
        return dense_tracks, {"source": "dense_export_fallback", "lossy": True}
    result = load_json(result_path)
    global_frames = {int(frame) for frame in result.get("non_empty_frames", [])}
    if not global_frames:
        return dense_tracks, {"source": "dense_export_fallback", "lossy": True}
    interval_map = result.get("segment_gating", {}).get("intervals", {})
    recovered: dict[int, dict[int, list[float]]] = {}
    counts: dict[str, int] = {}
    for object_id, track in dense_tracks.items():
        intervals = interval_map.get(str(object_id), interval_map.get(object_id, []))
        allowed = global_frames
        if intervals:
            allowed = {
                frame for frame in global_frames
                if any(int(lo) <= frame <= int(hi) for lo, hi in intervals)
            }
        recovered[object_id] = {
            frame: box for frame, box in track.items() if frame in allowed
        }
        counts[str(object_id)] = len(recovered[object_id])
    return recovered, {
        "source": "run_result.non_empty_frames+segment_gating.intervals",
        "lossy": True,
        "note": "target-specific raw SAM failures cannot be recovered from compact JSON",
        "recovered_frames_per_target": counts,
    }


def load_sparse_tracks(path: Path) -> dict[int, dict[int, list[float]]]:
    payload = load_json(path)
    result: dict[int, dict[int, list[float]]] = {}
    for frame_text, objects in payload["predictions_by_frame"].items():
        frame_idx = int(frame_text)
        for item in objects:
            object_id = int(item["object_id"])
            result.setdefault(object_id, {})[frame_idx] = sane_yxyx(item["box_2d_yxyx_1000"])
    return result


def plan_anchor_track(plan: dict[str, Any], object_id: int) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for anchor in plan.get("anchors", []):
        for box in anchor.get("boxes", []):
            if int(box["obj_id"]) == object_id:
                result[int(anchor["frame_idx"])] = sane_yxyx(box["box_2d"])
    return result


def anchor_alignment(plan: dict[str, Any], object_id: int,
                     dense: dict[int, list[float]]) -> tuple[float, int | None]:
    candidates = [
        (box_iou(dense[frame_idx], box), frame_idx)
        for frame_idx, box in plan_anchor_track(plan, object_id).items()
        if frame_idx in dense
    ]
    return max(candidates, default=(0.0, None))


def style_couple(
    dense: dict[int, list[float]],
    sparse: dict[int, list[float]],
    *,
    agreement_iou: float = 0.30,
    nearest_residual_frames: int = 60,
) -> tuple[dict[int, list[float]], list[int]]:
    """Apply v13's sparse-minus-dense coordinate residual to dense boxes."""
    anchors = sorted(
        frame_idx for frame_idx in sparse
        if frame_idx in dense and box_iou(dense[frame_idx], sparse[frame_idx]) >= agreement_iou
    )
    residual = {
        frame_idx: [sparse[frame_idx][axis] - dense[frame_idx][axis] for axis in range(4)]
        for frame_idx in anchors
    }
    output: dict[int, list[float]] = {}
    for frame_idx, box in dense.items():
        if frame_idx in residual:
            output[frame_idx] = list(sparse[frame_idx])
            continue
        previous = max((anchor for anchor in anchors if anchor < frame_idx), default=None)
        following = min((anchor for anchor in anchors if anchor > frame_idx), default=None)
        offset: list[float] | None
        if previous is not None and following is not None:
            weight = (frame_idx - previous) / (following - previous)
            offset = [
                residual[previous][axis] * (1.0 - weight) + residual[following][axis] * weight
                for axis in range(4)
            ]
        elif previous is not None and frame_idx - previous <= nearest_residual_frames:
            offset = residual[previous]
        elif following is not None and following - frame_idx <= nearest_residual_frames:
            offset = residual[following]
        else:
            offset = None
        if offset is None:
            output[frame_idx] = list(box)
            continue
        candidate = [box[axis] + offset[axis] for axis in range(4)]
        try:
            output[frame_idx] = sane_yxyx(candidate, min_size=2.0)
        except ValueError:
            output[frame_idx] = list(box)
    return output, anchors


def match_grid_oid(
    sam_track: dict[int, list[float]],
    sparse_tracks: dict[int, dict[int, list[float]]],
) -> tuple[int | None, float]:
    best_id, best_iou = None, 0.0
    for grid_id, grid_track in sparse_tracks.items():
        shared = sorted(set(sam_track) & set(grid_track))
        if len(shared) < 2:
            continue
        mean_iou = statistics.mean(
            box_iou(sam_track[frame], grid_track[frame]) for frame in shared
        )
        if mean_iou > best_iou:
            best_id, best_iou = grid_id, mean_iou
    return best_id, best_iou


def interpolate_track(
    track: dict[int, list[float]],
    short_gap: int,
    *,
    stable_iou: float = 0.30,
) -> dict[int, list[float]]:
    """Exact v13 rule: fill short gaps or long gaps with stable brackets."""
    frames = sorted(track)
    output = dict(track)
    for previous, following in zip(frames, frames[1:]):
        gap = following - previous
        if gap <= 1:
            continue
        if gap > short_gap and box_iou(track[previous], track[following]) < stable_iou:
            continue
        for frame in range(previous + 1, following):
            weight = (frame - previous) / gap
            output[frame] = [
                track[previous][axis] * (1.0 - weight)
                + track[following][axis] * weight
                for axis in range(4)
            ]
    return output


def infer_num_frames(sample_dir: Path, dense_tracks: dict[int, dict[int, list[float]]]) -> int:
    result_path = sample_dir / "run_result.json"
    if result_path.is_file():
        value = int(load_json(result_path).get("num_video_frames", 0))
        if value > 0:
            return value
    maximum = max((max(track) for track in dense_tracks.values() if track), default=-1)
    if maximum < 0:
        raise ValueError(f"cannot infer video length from {sample_dir}")
    return maximum + 1


def direct_id_mapping(plan_ids: list[int], dense_ids: set[int], sparse_ids: set[int]) -> None:
    """Fail loudly instead of silently coupling boxes from different identities."""
    unknown = (dense_ids | sparse_ids) - set(plan_ids)
    if unknown:
        raise ValueError(f"track IDs absent from saved plan: {sorted(unknown)}")


def official_payload(video_id: str, question_id: str,
                     tracks: dict[int, dict[int, list[float]]]) -> dict[str, Any]:
    official_tracks = []
    for object_id, track in sorted(tracks.items()):
        frames = sorted(track)
        official_tracks.append({
            "id": object_id,
            "score": 1.0,
            "bounding_boxes": [yxyx_1000_to_xyxy_normalized(track[frame]) for frame in frames],
            "frame_ids": frames,
        })
    return {video_id: {"grounded_question": {str(question_id): official_tracks}}}


def fuse_sample(args: argparse.Namespace, video_id: str, question_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    sample = sample_name(video_id, question_id)
    plan_path = args.plan_root / sample / "multi_anchor_plan.json"
    dense_dir = args.dense_root / sample
    dense_path = dense_dir / "predictions.json"
    sparse_path = args.sparse_root / sample / "sparse_predictions.json"
    for path in (plan_path, dense_path, sparse_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    plan = load_json(plan_path)
    plan_ids = plan_target_ids(plan)
    dense_full = load_dense_tracks(dense_path, video_id, question_id)
    dense_tracks, coverage_meta = recover_sampled_sam_tracks(dense_dir, dense_full)
    sparse_tracks = load_sparse_tracks(sparse_path)
    direct_id_mapping(plan_ids, set(dense_tracks), set(sparse_tracks))
    num_frames = infer_num_frames(dense_dir, dense_full)

    sam_by_grid: dict[int, dict[int, list[float]]] = {}
    gate_by_grid: dict[int, dict[str, Any]] = {}
    sam_candidates: list[dict[str, Any]] = []
    for sam_id, sam_track in dense_tracks.items():
        grid_id = sam_id
        grid_track = sparse_tracks.get(grid_id, {})
        shared = sorted(set(sam_track) & set(grid_track))
        rematched = False
        if len(shared) < args.min_shared:
            matched_id, _ = match_grid_oid(sam_track, sparse_tracks)
            if matched_id is None:
                sam_candidates.append({
                    "sam_object_id": sam_id, "trusted": False,
                    "reason": "no_grid_match", "num_shared_sparse_frames": len(shared),
                })
                continue
            grid_id = matched_id
            grid_track = sparse_tracks[grid_id]
            shared = sorted(set(sam_track) & set(grid_track))
            rematched = grid_id != sam_id
        correlation = (
            statistics.mean(box_iou(sam_track[frame], grid_track[frame]) for frame in shared)
            if shared else 0.0
        )
        seed_iou, seed_frame = anchor_alignment(plan, sam_id, sam_track)
        trusted = (
            seed_iou >= args.min_anchor_iou
            and len(shared) >= args.min_shared
            and correlation >= args.gate_iou
        )
        gate = {
            "sam_object_id": sam_id, "grid_object_id": grid_id,
            "id_rematched": rematched, "trusted": trusted,
            "seed_alignment_iou": round(seed_iou, 6),
            "seed_frame_idx": seed_frame,
            "num_shared_sparse_frames": len(shared),
            "mean_sparse_dense_iou": round(correlation, 6),
        }
        sam_candidates.append(gate)
        previous_gate = gate_by_grid.get(grid_id)
        if previous_gate is None or correlation > previous_gate["mean_sparse_dense_iou"]:
            gate_by_grid[grid_id] = gate
        if trusted:
            sam_by_grid[grid_id] = sam_track

    final_tracks: dict[int, dict[int, list[float]]] = {}
    target_meta: list[dict[str, Any]] = []
    for grid_id, grid_track in sorted(sparse_tracks.items()):
        merged = dict(grid_track)
        coupling_anchors: list[int] = []
        trusted = grid_id in sam_by_grid
        if trusted:
            # v13 keeps the agreement threshold fixed at 0.30 even when the
            # outer trust gate is tuned separately.
            styled, coupling_anchors = style_couple(
                sam_by_grid[grid_id], grid_track, agreement_iou=0.30
            )
            merged.update(styled)
        final = interpolate_track(merged, args.short_gap)
        if final:
            final_tracks[grid_id] = final
        gate = gate_by_grid.get(grid_id, {})
        target_meta.append({
            "object_id": grid_id,
            "trusted": trusted,
            "owner": "sparse_grid+trusted_sampled_sam" if trusted else "sparse_grid_only",
            "sam_object_id": gate.get("sam_object_id"),
            "seed_alignment_iou": gate.get("seed_alignment_iou", 0.0),
            "seed_frame_idx": gate.get("seed_frame_idx"),
            "num_shared_sparse_frames": gate.get("num_shared_sparse_frames", 0),
            "mean_sparse_dense_iou": gate.get("mean_sparse_dense_iou", 0.0),
            "coupling_anchor_frames": coupling_anchors,
            "num_output_frames": len(final),
        })

    payload = official_payload(video_id, question_id, final_tracks)
    meta = {
        "sample": sample, "status": "complete", "num_video_frames": num_frames,
        "fusion_mode": "v13_sparse_first_sampled_sam_overlay",
        "sam_coverage_recovery": coverage_meta,
        "thresholds": {
            "min_anchor_iou": args.min_anchor_iou,
            "min_shared": args.min_shared,
            "gate_iou": args.gate_iou,
            "style_agreement_iou": 0.30,
            "short_gap": args.short_gap,
            "long_gap_bracket_iou": 0.30,
        },
        "sam_candidates": sam_candidates,
        "targets": target_meta,
    }
    return payload, meta


def validate_official_sample(payload: dict[str, Any], video_id: str,
                             question_id: str, num_frames: int) -> None:
    tracks = payload[video_id]["grounded_question"][str(question_id)]
    if len(tracks) > 10:
        raise ValueError("challenge accepts at most 10 tracks")
    for track in tracks:
        frames, boxes = track["frame_ids"], track["bounding_boxes"]
        if frames != sorted(set(frames)):
            raise ValueError(f"track {track['id']} frame_ids are not strictly increasing")
        if any(frame < 0 or frame >= num_frames for frame in frames):
            raise ValueError(f"track {track['id']} has out-of-range frame_ids")
        if len(frames) != len(boxes):
            raise ValueError(f"track {track['id']} frame/box mismatch")
        for box in boxes:
            if len(box) != 4 or not all(0.0 <= float(value) <= 1.0 for value in box):
                raise ValueError(f"track {track['id']} invalid normalized box {box}")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"track {track['id']} degenerate box {box}")


def merge_sample(merged: dict[str, Any], sample_payload: dict[str, Any]) -> None:
    for video_id, video_payload in sample_payload.items():
        destination = merged.setdefault(video_id, {"grounded_question": {}})
        questions = video_payload["grounded_question"]
        overlap = set(destination["grounded_question"]) & set(questions)
        if overlap:
            raise ValueError(f"duplicate merged questions for {video_id}: {sorted(overlap)}")
        destination["grounded_question"].update(questions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--sparse-root", type=Path, default=DEFAULT_SPARSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--min-anchor-iou", type=float, default=0.25)
    parser.add_argument("--min-shared", type=int, default=3)
    parser.add_argument("--gate-iou", type=float, default=0.30)
    parser.add_argument("--short-gap", type=int, default=6)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", action="append")
    args = parser.parse_args()
    for name in ("plan_root", "dense_root", "sparse_root", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    args = parse_args()
    rows = select_rows(
        load_subset(args.subset), samples=set(args.sample) if args.sample else None,
        num_shards=args.num_shards, shard_index=args.shard_index, limit=args.limit,
    )
    merged: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for video_id, question_id in rows:
        sample = sample_name(video_id, question_id)
        try:
            payload, meta = fuse_sample(args, video_id, question_id)
            validate_official_sample(payload, video_id, question_id, meta["num_video_frames"])
            sample_dir = args.output_root / sample
            save_json(sample_dir / "predictions.json", payload)
            save_json(sample_dir / "fusion_meta.json", meta)
            merge_sample(merged, payload)
            summaries.append(meta)
            trusted = sum(target["trusted"] for target in meta["targets"])
            print(f"complete {sample}: trusted={trusted}/{len(meta['targets'])}", flush=True)
        except Exception as exc:
            failures.append({"sample": sample, "error_type": type(exc).__name__, "error": str(exc)})
            print(f"FAILED {sample}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    save_json(args.output_root / f"predictions_shard_{args.shard_index:03d}.json", merged)
    owners: dict[str, int] = {}
    for summary in summaries:
        for target in summary["targets"]:
            owners[target["owner"]] = owners.get(target["owner"], 0) + 1
    report = {
        "num_selected": len(rows), "num_complete": len(summaries), "num_failed": len(failures),
        "target_owner_counts": owners, "samples": summaries, "failures": failures,
    }
    save_json(args.output_root / f"fusion_summary_shard_{args.shard_index:03d}.json", report)
    if args.num_shards == 1 and not failures:
        save_json(args.output_root / "predictions.json", merged)
        print(f"submission candidate: {args.output_root / 'predictions.json'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
