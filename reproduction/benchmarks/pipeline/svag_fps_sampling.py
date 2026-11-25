#!/usr/bin/env python3
"""Select SVAG anchor candidates by elapsed time in the original video.

This module has no model, video-decoder, annotation, or third-party dependency.
The low-FPS exception retains every original frame so short appearances in
5-FPS OVIS clips do not disappear between anchor candidates. Higher-FPS videos
use rounded timestamp samples, plus both endpoints. Returned values always
refer to original frame indices, rather than planning-proxy frame numbers.
"""
from __future__ import annotations

import math
from numbers import Integral, Real


def _finite_rate(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number')
    rate = float(value)
    if not math.isfinite(rate) or rate < 0 or (not allow_zero and rate == 0):
        condition = 'non-negative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be finite and {condition}')
    return rate


def sample_anchor_frames(
    num_frames: int,
    source_fps: float,
    target_fps: float = 3.0,
    low_fps_full_threshold: float = 5.0,
) -> list[int]:
    """Return deterministic, strictly increasing original-frame candidates.

    At or below ``low_fps_full_threshold``, every original frame is eligible.
    Above that threshold, sample timestamps at ``min(source_fps, target_fps)``
    and round to the closest original index. Always retain frame 0 and the last
    frame, even when the last frame is off the regular timestamp grid. Set the
    threshold to zero to disable the low-FPS exception.

    This is a candidate grid, not a guarantee that it intersects every possible
    one-frame appearance in a high-FPS video. A planner may add exact original
    frames inside a predicted appearance interval when that interval has no
    candidate; such additions must also reach the SAM input frame map.
    """
    if isinstance(num_frames, bool) or not isinstance(num_frames, Integral) or num_frames < 1:
        raise ValueError('num_frames must be a positive integer')
    source_rate = _finite_rate(source_fps, 'source_fps')
    target_rate = _finite_rate(target_fps, 'target_fps')
    threshold = _finite_rate(low_fps_full_threshold, 'low_fps_full_threshold', allow_zero=True)
    count = int(num_frames)
    if source_rate <= threshold:
        return list(range(count))

    effective_rate = min(source_rate, target_rate)
    frame_step = source_rate / effective_rate
    last = count - 1
    sample_count = int(math.floor(last / frame_step)) + 1
    selected = {0, last}
    selected.update(
        index
        for sample in range(sample_count)
        if 0 <= (index := round(sample * frame_step)) < count
    )
    return sorted(selected)
