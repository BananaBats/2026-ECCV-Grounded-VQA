#!/usr/bin/env python3
"""Pure source-frame maps for saved-plan SVAG tracking at up to 10 FPS.

The timestamp grid is authoritative; ``sam_stride`` is a compatibility label
and cannot represent the alternating intervals of 25/14 FPS source videos.
Exact saved anchors and dense visibility-boundary windows are added without
altering the plan. Semantic answer/target coverage stays with plan validation.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


METHOD_REVISION = 'svag-sam3-fps10-v1'


def _rate(value: Any, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number')
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not zero and result == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return result


def _frame(value: Any, name: str, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < count:
        raise ValueError(f'{name} must be an original frame index in [0, {count-1}]')
    return int(value)


def _identity(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _validate_anchor(anchor: dict[str, Any], count: int) -> int:
    if not isinstance(anchor, dict):
        raise ValueError('Each anchor must be an object')
    frame = _frame(anchor.get('frame_idx'), 'anchor.frame_idx', count)
    obj_ids = anchor.get('obj_ids')
    if obj_ids is not None:
        if not isinstance(obj_ids, list):
            raise ValueError('anchor.obj_ids must be a list')
        ids = [_identity(value, 'anchor.obj_ids') for value in obj_ids]
        if len(ids) != len(set(ids)):
            raise ValueError('anchor.obj_ids contains duplicate IDs')
    boxes = anchor.get('boxes')
    if boxes is not None:
        if not isinstance(boxes, list):
            raise ValueError('anchor.boxes must be a list')
        box_ids = []
        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError('Each anchor box must be an object')
            box_ids.append(_identity(box.get('obj_id'), 'box.obj_id'))
            coords = box.get('box_2d')
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                raise ValueError('box_2d must contain four YXYX coordinates')
            if any(isinstance(x, bool) or not isinstance(x, Real)
                   or not math.isfinite(x) or not 0 <= x <= 1000 for x in coords):
                raise ValueError('box_2d must be finite normalized coordinates in [0,1000]')
            y0, x0, y1, x1 = coords
            if y0 >= y1 or x0 >= x1:
                raise ValueError('box_2d must have positive area')
        if len(box_ids) != len(set(box_ids)):
            raise ValueError('anchor.boxes contains duplicate object IDs')
        if obj_ids is not None and set(box_ids) != set(obj_ids):
            raise ValueError('anchor.boxes IDs differ from anchor.obj_ids')
    return frame


def build_tracking_map(
    plan: dict[str, Any], num_frames: int, source_fps: float,
    target_fps: float = 10., boundary_seconds: float = .5,
    gate_seconds: float = .5,
) -> dict[str, Any]:
    """Return an exact original-index tracking schedule without mutating plan.

    Source videos at or below target_fps retain every frame. Other videos use
    round(k*source_fps/target_fps), plus exact saved anchors, visibility-boundary
    windows, and the last source frame. Old ``anchor.sam_frame_idx`` values are
    deliberately ignored and rebuilt in ``anchor_sam_indices``.
    """
    if not isinstance(plan, dict):
        raise ValueError('plan must be an object')
    if isinstance(num_frames, bool) or not isinstance(num_frames, Integral) or num_frames < 1:
        raise ValueError('num_frames must be a positive integer')
    count = int(num_frames)
    source = _rate(source_fps, 'source_fps')
    requested = _rate(target_fps, 'target_fps')
    boundary = _rate(boundary_seconds, 'boundary_seconds', zero=True)
    gate = _rate(gate_seconds, 'gate_seconds', zero=True)
    effective = min(source, requested)
    low = source <= requested
    if low:
        base = list(range(count))
    else:
        timestamp_count = math.floor((count-1)/source*effective)+1
        base = sorted({min(count-1, round(k*source/effective)) for k in range(timestamp_count)})
    anchors = plan.get('anchors', [])
    targets = plan.get('targets', [])
    if not isinstance(anchors, list) or not isinstance(targets, list):
        raise ValueError('plan anchors and targets must be lists')
    anchor_frames = [_validate_anchor(anchor, count) for anchor in anchors]
    if len(anchor_frames) != len(set(anchor_frames)):
        raise ValueError('Saved plan contains duplicate anchor frames')
    boundaries = set()
    for key in ['first_visible_frame_idx', 'last_visible_frame_idx']:
        if key in plan:
            boundaries.add(_frame(plan[key], key, count))
    if ('first_visible_frame_idx' in plan and 'last_visible_frame_idx' in plan
            and plan['first_visible_frame_idx'] > plan['last_visible_frame_idx']):
        raise ValueError('Global visibility bounds are reversed')
    segment_count = 0
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError('Each target must be an object')
        segments = target.get('visibility_segments', [])
        if not isinstance(segments, list):
            raise ValueError('visibility_segments must be a list')
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError('Each visibility segment must be an object')
            first = _frame(segment.get('first_frame_idx'), 'visibility.first_frame_idx', count)
            last = _frame(segment.get('last_frame_idx'), 'visibility.last_frame_idx', count)
            if first > last:
                raise ValueError('Visibility segment bounds are reversed')
            boundaries.update([first, last])
            segment_count += 1
    radius = math.ceil(source*boundary)
    frames = set(base) | set(anchor_frames) | {count-1}
    for frame in boundaries:
        frames.update(range(max(0, frame-radius), min(count-1, frame+radius)+1))
    sampled = sorted(frames)
    source_to_sam = {frame: index for index, frame in enumerate(sampled)}
    return {
        'method_revision': METHOD_REVISION,
        'sampling_mode': 'timestamp_fps_plus_exact_anchors_and_visibility_boundaries',
        'num_original_frames': count, 'source_fps': source,
        'requested_target_fps': requested, 'effective_target_fps': effective,
        'sam_target_fps': effective,
        'sam_stride': max(1, round(source/effective)),
        'required_frame_stride': 0,
        'boundary_seconds': boundary, 'gate_seconds': gate,
        'boundary_radius_seconds': boundary, 'segment_gate_margin_seconds': gate,
        'boundary_radius': radius, 'segment_gate_margin': math.ceil(source*gate),
        'low_fps_all_frames': low,
        'base_source_indices': base, 'num_base_sampled_frames': len(base),
        'num_base_stride_frames': len(base),
        'segment_boundaries': sorted(boundaries), 'num_visibility_segments': segment_count,
        'anchor_source_indices': sorted(anchor_frames), 'num_anchor_frames': len(anchor_frames),
        'sampled_frame_ids': sampled, 'num_sam_frames': len(sampled),
        'num_additional_frames': len(sampled)-len(base),
        'sam_to_original_frame': {str(index): frame for index, frame in enumerate(sampled)},
        'anchor_sam_indices': {str(frame): source_to_sam[frame] for frame in sorted(anchor_frames)},
        'timestamp_formula': 'round(k * source_fps / min(source_fps, target_fps))',
    }
