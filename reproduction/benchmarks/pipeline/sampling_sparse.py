"""Original GT-free SVAG sparse grid and saved-anchor coverage additions."""
import math

def native_grid(num_frames, source_fps, target_fps=1.):
    if type(num_frames) is not int or num_frames < 1:
        raise ValueError('num_frames must be a positive integer')
    for name, value in [('source_fps', source_fps), ('target_fps', target_fps)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    effective = min(source_fps, target_fps)
    count = math.floor((num_frames-1)/source_fps*effective)+1
    return sorted({min(num_frames-1, round(k*source_fps/effective)) for k in range(count)})

def _visible(target, index):
    segments = target.get('visibility_segments') or []
    return not segments or any(segment['first_frame_idx'] <= index <= segment['last_frame_idx']
                               for segment in segments)

def coverage_schedule(plan, base):
    """Add only saved, target-specific anchors when the regular grid misses it."""
    additions, reasons, warnings = set(), [], []
    for target in plan['targets']:
        if any(_visible(target, frame) for frame in base):
            continue
        identity = target['obj_id']
        choices = sorted({int(anchor['frame_idx']) for anchor in plan.get('anchors', [])
                          if _visible(target, int(anchor['frame_idx'])) and
                          any(box['obj_id'] == identity for box in anchor.get('boxes', []))})
        if choices:
            frame = choices[0]; additions.add(frame)
            reasons.append({'object_id': identity, 'frame_idx': frame,
                            'reason': 'No visibly confirmed 1Hz grid frame; use this target\'s earliest existing visible saved anchor.'})
        else:
            warnings.append({'object_id': identity,
                             'reason': 'No visibly confirmed grid frame or usable saved anchor; no frame/box invented.'})
    return sorted(set(base)|additions), sorted(additions), reasons, warnings
