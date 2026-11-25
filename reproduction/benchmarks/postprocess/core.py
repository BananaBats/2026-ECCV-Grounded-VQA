#!/usr/bin/env python3
"""Fuse saved sparse boxes and sampled SAM boxes with one IoU threshold.

For each saved sparse identity, use the SAM track with the same stable ID.
Shared boxes with IoU >= tau define sparse-minus-SAM coordinate residuals.
Interpolate these residuals between observations and extend the nearest residual
outside them; if none qualify, keep the raw SAM boxes. Corrected SAM boxes own
their available frames and sparse boxes fill the remaining frames. A remaining
gap is interpolated only when its endpoint boxes have IoU >= the SAME tau.

There is no anchor-alignment gate, mean-agreement gate, count gate, ID rematch,
short-gap exception, or residual carry cutoff. Inputs are saved inference
artifacts only: this module does not load annotations, run models, or use GPUs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from .legacy.sparse_common import (box_iou, load_json, load_subset, plan_target_ids,
                           sample_name, sane_yxyx, save_json)
from .legacy.fuse_dense_sparse import (direct_id_mapping, infer_num_frames,
    load_dense_tracks, load_sparse_tracks, merge_sample, official_payload,
    recover_sampled_sam_tracks, validate_official_sample)


METHOD_REVISION = 'minimal-residual-one-iou-v1'


def validate_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('iou_threshold must be a finite value in [0, 1]')
    return float(value)


def correct_sam_track(
    sam: dict[int, list[float]], sparse: dict[int, list[float]],
    iou_threshold: float = 0.30,
) -> tuple[dict[int, list[float]], list[int]]:
    """Transfer sparse coordinate residuals with unlimited nearest extension."""
    tau = validate_threshold(iou_threshold)
    anchors = sorted(frame for frame in sparse
                     if frame in sam and box_iou(sam[frame], sparse[frame]) >= tau)
    residuals = {frame: [sparse[frame][axis]-sam[frame][axis] for axis in range(4)]
                 for frame in anchors}
    corrected: dict[int, list[float]] = {}
    for frame, box in sam.items():
        if frame in residuals:
            corrected[frame] = list(sparse[frame])
            continue
        index = bisect_left(anchors, frame)
        previous = anchors[index-1] if index else None
        following = anchors[index] if index < len(anchors) else None
        if previous is not None and following is not None:
            weight = (frame-previous)/(following-previous)
            offset = [residuals[previous][axis]*(1.0-weight)
                      + residuals[following][axis]*weight for axis in range(4)]
        elif previous is not None:
            offset = residuals[previous]
        elif following is not None:
            offset = residuals[following]
        else:
            corrected[frame] = list(box)
            continue
        candidate = [box[axis]+offset[axis] for axis in range(4)]
        try:
            # Preserve the original geometric validity check, not a trust gate.
            corrected[frame] = sane_yxyx(candidate, min_size=2.0)
        except ValueError:
            corrected[frame] = list(box)
    return corrected, anchors


def interpolate_stable_gaps(
    track: dict[int, list[float]], iou_threshold: float = 0.30,
) -> dict[int, list[float]]:
    """Fill a gap iff its endpoint boxes pass the same IoU threshold."""
    tau = validate_threshold(iou_threshold)
    frames = sorted(track)
    result = dict(track)
    for previous, following in zip(frames, frames[1:]):
        gap = following-previous
        if gap <= 1 or box_iou(track[previous], track[following]) < tau:
            continue
        for frame in range(previous+1, following):
            weight = (frame-previous)/gap
            result[frame] = [track[previous][axis]*(1.0-weight)
                             + track[following][axis]*weight for axis in range(4)]
    return result


def fuse_tracks(
    sam_tracks: dict[int, dict[int, list[float]]],
    sparse_tracks: dict[int, dict[int, list[float]]],
    iou_threshold: float = 0.30,
) -> tuple[dict[int, dict[int, list[float]]], list[dict[str, Any]]]:
    """Fuse each sparse ID with exactly its same-ID SAM track, without rematch."""
    tau = validate_threshold(iou_threshold)
    final: dict[int, dict[int, list[float]]] = {}
    targets = []
    for object_id, sparse in sorted(sparse_tracks.items()):
        sam = sam_tracks.get(object_id, {})
        corrected, anchors = correct_sam_track(sam, sparse, tau)
        merged = dict(sparse)
        merged.update(corrected)
        fused = interpolate_stable_gaps(merged, tau)
        if fused:
            final[object_id] = fused
        targets.append({'object_id': object_id, 'same_id_sam_available': bool(sam),
            'residual_anchor_frames': anchors, 'num_sam_frames': len(sam),
            'num_sparse_frames': len(sparse), 'num_output_frames': len(fused),
            'raw_sam_without_residual': bool(sam) and not anchors})
    return final, targets


def fuse_sample(args, video_id: str, question_id: str):
    sample = sample_name(video_id, question_id)
    plan_path = args.plan_root/sample/'multi_anchor_plan.json'
    dense_directory = args.dense_root/sample
    dense_path = dense_directory/'predictions.json'
    sparse_path = args.sparse_root/sample/'sparse_predictions.json'
    plan = load_json(plan_path)
    dense_full = load_dense_tracks(dense_path, video_id, question_id)
    sampled_sam, recovery = recover_sampled_sam_tracks(dense_directory, dense_full)
    sparse = load_sparse_tracks(sparse_path)
    direct_id_mapping(plan_target_ids(plan), set(sampled_sam), set(sparse))
    num_frames = infer_num_frames(dense_directory, dense_full)
    tracks, targets = fuse_tracks(sampled_sam, sparse, args.iou_threshold)
    payload = official_payload(video_id, question_id, tracks)
    validate_official_sample(payload, video_id, question_id, num_frames)
    metadata = {'sample': sample, 'status': 'complete', 'method_revision': METHOD_REVISION,
        'iou_threshold': args.iou_threshold, 'num_video_frames': num_frames,
        'identity_matching': 'same_global_plan_id_without_rematching',
        'sam_coverage_recovery': recovery, 'targets': targets,
        'sam_ids_without_sparse_track': sorted(set(sampled_sam)-set(sparse)),
        'source_inputs': {'plan': str(plan_path), 'dense': str(dense_path), 'sparse': str(sparse_path)}}
    return payload, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['subset', 'plan-root', 'dense-root', 'sparse-root', 'output-root']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.30)
    args = parser.parse_args(argv)
    args.iou_threshold = validate_threshold(args.iou_threshold)
    for name in ['subset', 'plan_root', 'dense_root', 'sparse_root', 'output_root']:
        setattr(args, name, getattr(args, name).resolve())
    rows = load_subset(args.subset)
    if not rows or len(rows) != len(set(rows)):
        parser.error('Subset must contain unique nonempty video/question rows')
    merged, summaries, failures = {}, [], []
    for video_id, question_id in rows:
        sample = sample_name(video_id, question_id)
        try:
            payload, metadata = fuse_sample(args, video_id, question_id)
            directory = args.output_root/sample
            save_json(directory/'predictions.json', payload)
            save_json(directory/'fusion_meta.json', metadata)
            merge_sample(merged, payload)
            summaries.append(metadata)
            print(f'complete {sample}', flush=True)
        except Exception as error:
            failures.append({'sample': sample, 'error_type': type(error).__name__, 'error': str(error)})
            print(f'FAILED {sample}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
    summary = {'method_revision': METHOD_REVISION, 'iou_threshold': args.iou_threshold,
        'num_selected': len(rows), 'num_complete': len(summaries), 'num_failed': len(failures),
        'model_calls': 0, 'GPU_calls': 0, 'annotation_inputs': [],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'samples': summaries, 'failures': failures}
    save_json(args.output_root/'fusion_summary.json', summary)
    if not failures:
        save_json(args.output_root/'predictions.json', merged)
        print(f'submission candidate: {args.output_root / "predictions.json"}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
