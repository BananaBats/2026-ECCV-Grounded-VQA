"""Minimal SD-Track fusion in source-frame coordinates (YXYX on a 0 to 1000 scale)."""
from __future__ import annotations

from bisect import bisect_left
import math
from typing import Iterable


def sane_yxyx(box: Iterable[float], *, min_size: float = 1.0) -> list[float]:
    values = [float(value) for value in box]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid box: {values}")
    y0, x0, y1, x1 = (min(1000.0, max(0.0, value)) for value in values)
    if y1 - y0 < min_size or x1 - x0 < min_size:
        raise ValueError(f"degenerate yxyx box: {values}")
    return [y0, x0, y1, x1]


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ay0, ax0, ay1, ax1 = [float(value) for value in a]
    by0, bx0, by1, bx1 = [float(value) for value in b]
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    intersection = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def yxyx_1000_to_xyxy_normalized(box: Iterable[float]) -> list[float]:
    y0, x0, y1, x1 = sane_yxyx(box)
    return [x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0]


IOU_THRESHOLD = 0.30


def interpolate(left, right, weight):
    return [a * (1.0 - weight) + b * weight for a, b in zip(left, right)]


def correct_sam_track(sam, sparse):
    """Transfer sparse-minus-dense offsets to existing SAM3 boxes."""
    references = sorted(frame for frame in sparse if frame in sam
                        and box_iou(sam[frame], sparse[frame]) >= IOU_THRESHOLD)
    if not references:
        return dict(sam)
    offsets = {f: [a - b for a, b in zip(sparse[f], sam[f])] for f in references}
    corrected = {}
    for frame, box in sam.items():
        if frame in offsets:
            corrected[frame] = list(sparse[frame])
            continue
        index = bisect_left(references, frame)
        if index == 0:
            offset = offsets[references[0]]
        elif index == len(references):
            offset = offsets[references[-1]]
        else:
            left, right = references[index - 1], references[index]
            weight = (frame - left) / (right - left)
            offset = interpolate(offsets[left], offsets[right], weight)
        candidate = [coordinate + delta for coordinate, delta in zip(box, offset)]
        try:
            corrected[frame] = sane_yxyx(candidate, min_size=2.0)
        except ValueError:
            corrected[frame] = list(box)
    return corrected


def interpolate_stable_gaps(track):
    """Fill source frames between overlapping endpoint boxes."""
    frames = sorted(track)
    result = dict(track)
    for left, right in zip(frames, frames[1:]):
        if right == left + 1 or box_iou(track[left], track[right]) < IOU_THRESHOLD:
            continue
        for frame in range(left + 1, right):
            weight = (frame - left) / (right - left)
            result[frame] = interpolate(track[left], track[right], weight)
    return result


def fuse(query, plan, sam, gemini):
    """Emit normalized XYXY tracks in the Perception Test prediction format."""
    target_ids = {int(t["obj_id"]) for t in plan["targets"]}
    if (set(sam) | set(gemini)) - target_ids:
        raise ValueError("An observation has an identity missing from the plan")
    tracks = []
    for identity, sparse_track in sorted(gemini.items()):
        corrected = correct_sam_track(sam.get(identity, {}), sparse_track)
        merged = {**sparse_track, **corrected}
        filled = interpolate_stable_gaps(merged)
        if filled:
            frames = sorted(filled)
            tracks.append({
                "id": identity,
                "score": 1.0,
                "frame_ids": frames,
                "bounding_boxes": [yxyx_1000_to_xyxy_normalized(filled[i]) for i in frames],
            })
    return {query["video_id"]: {"grounded_question": {str(query["question_id"]): tracks}}}
