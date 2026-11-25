"""Pure, GT-free repair of saved SVAG planner structure.

Target identities and predicted time intervals are never rewritten.  The helper
can add initialization frames from those predicted visibility intervals, but it
does not fabricate spatial boxes.  Different boxes for one identity must be
resolved by the caller, normally with a fresh grounding request.
"""
from __future__ import annotations

from copy import deepcopy
import math


SOFT_ANCHOR_LIMIT = 3


class NeedsTemporalRepair(ValueError):
    """Saved action intervals extend outside the same actor's visibility."""

    def __init__(self, obj_id, action_segment):
        self.obj_id = obj_id
        self.action_segment = deepcopy(action_segment)
        super().__init__(
            f'obj_id={obj_id}: action segment {action_segment} is not covered '
            'by its predicted visibility segments; request temporal repair')


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _segments(target, key, num_frames):
    value = target.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f'obj_id={target.get("obj_id")}: {key} must be nonempty')
    for segment in value:
        if not isinstance(segment, dict):
            raise ValueError(f'{key} must contain segment dictionaries')
        first = _integer(segment.get('first_frame_idx'), f'{key}.first_frame_idx')
        last = _integer(segment.get('last_frame_idx'), f'{key}.last_frame_idx')
        if first > last or last >= num_frames:
            raise ValueError(f'Invalid original-frame bounds in {key}: {segment}')
    return value


def _time_targets(raw, num_frames):
    targets = raw.get('targets')
    if not isinstance(targets, list) or not targets:
        raise ValueError('targets must be a nonempty list')
    ids = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError('targets must contain dictionaries')
        obj_id = _integer(target.get('obj_id'), 'obj_id')
        ids.append(obj_id)
        for field in ('object_label', 'visual_identity_description'):
            if not isinstance(target.get(field), str) or not target[field].strip():
                raise ValueError(f'obj_id={obj_id}: {field} must be nonempty text')
        visibility = _segments(target, 'visibility_segments', num_frames)
        actions = _segments(target, 'action_segments', num_frames)
        if not isinstance(target.get('action_reasoning'), str) or not target['action_reasoning'].strip():
            raise ValueError(f'obj_id={obj_id}: action_reasoning must be nonempty text')
        for action in actions:
            cursor = action['first_frame_idx']
            for visible in sorted(visibility, key=lambda s: s['first_frame_idx']):
                if visible['last_frame_idx'] < cursor:
                    continue
                if visible['first_frame_idx'] > cursor:
                    break
                cursor = visible['last_frame_idx'] + 1
                if cursor > action['last_frame_idx']:
                    break
            if cursor <= action['last_frame_idx']:
                raise NeedsTemporalRepair(obj_id, action)
    if sorted(ids) != list(range(len(ids))):
        raise ValueError('Target IDs must be unique and contiguous from zero; never renumber during recovery')
    return {target['obj_id']: target for target in targets}


def _visible(target, frame):
    return any(s['first_frame_idx'] <= frame <= s['last_frame_idx']
               for s in target['visibility_segments'])


def _margin(target, frame):
    """Depth inside a supported interval; larger values avoid its boundaries."""
    return max(min(frame-s['first_frame_idx'], s['last_frame_idx']-frame)
               for s in target['visibility_segments']
               if s['first_frame_idx'] <= frame <= s['last_frame_idx'])


def repair_anchor_selection(raw: dict, allowed_frames: list[int], num_frames: int):
    """Return (selection, audit changes), preserving every saved target.

    Existing anchors are snapped only to the supplied native-frame grid. Targets
    invisible at a snapped frame are removed from that anchor, then recovered at
    an existing supported anchor or a new frame. An exact midpoint outside the
    grid is introduced only when a target has no supported grid candidate.
    The old three-anchor cap becomes an explicitly recorded recovery exception.
    """
    _integer(num_frames, 'num_frames', minimum=1)
    if not isinstance(raw, dict):
        raise ValueError('selection must be a dictionary')
    if (not isinstance(allowed_frames, list) or not allowed_frames
            or any(type(f) is not int or not 0 <= f < num_frames for f in allowed_frames)
            or allowed_frames != sorted(set(allowed_frames))):
        raise ValueError('allowed_frames must be nonempty, sorted unique native-frame indices')
    result = deepcopy(raw)
    targets = _time_targets(result, num_frames)
    first = _integer(result.get('first_visible_frame_idx'), 'first_visible_frame_idx')
    last = _integer(result.get('last_visible_frame_idx'), 'last_visible_frame_idx')
    if first > last or last >= num_frames:
        raise ValueError('Invalid global visibility bounds')
    changes = []
    target_first = min(s['first_frame_idx'] for t in targets.values() for s in t['visibility_segments'])
    target_last = max(s['last_frame_idx'] for t in targets.values() for s in t['visibility_segments'])
    expanded = min(first, target_first), max(last, target_last)
    if expanded != (first, last):
        result['first_visible_frame_idx'], result['last_visible_frame_idx'] = expanded
        changes.append({'kind': 'global_visibility_expanded', 'from': [first, last], 'to': list(expanded)})
    raw_anchors = result.get('anchors')
    if not isinstance(raw_anchors, list):
        raise ValueError('anchors must be a list')
    by_frame = {}
    for anchor_index, anchor in enumerate(raw_anchors):
        if not isinstance(anchor, dict):
            raise ValueError('anchors must contain dictionaries')
        frame = _integer(anchor.get('frame_idx'), 'anchor.frame_idx')
        if frame >= num_frames:
            raise ValueError('Anchor is outside the source video')
        aligned = min(allowed_frames, key=lambda f: (abs(f-frame), f))
        if frame != aligned:
            changes.append({'kind': 'anchor_aligned_to_grid', 'from': frame, 'to': aligned})
        ids = anchor.get('obj_ids')
        if not isinstance(ids, list) or any(type(i) is not int or i not in targets for i in ids):
            raise ValueError('Anchor refers to an unknown target identity')
        good = sorted({i for i in ids if _visible(targets[i], aligned)})
        removed = sorted(set(ids)-set(good))
        if removed:
            changes.append({'kind': 'unsupported_anchor_memberships_removed', 'frame_idx': aligned, 'obj_ids': removed})
        if len(set(ids)) != len(ids):
            changes.append({'kind': 'duplicate_anchor_memberships_removed', 'frame_idx': aligned})
        if aligned in by_frame:
            by_frame[aligned]['obj_ids'] = sorted(set(by_frame[aligned]['obj_ids']) | set(good))
            changes.append({'kind': 'same_frame_anchors_merged', 'frame_idx': aligned, 'source_anchor_index': anchor_index})
        else:
            value = deepcopy(anchor)
            value.update(frame_idx=aligned, obj_ids=good)
            if not isinstance(value.get('reason'), str):
                raise ValueError('Anchor reason must be text')
            by_frame[aligned] = value

    covered = {i for a in by_frame.values() for i in a['obj_ids']}
    for obj_id in sorted(set(targets)-covered):
        candidates = [f for f in by_frame if _visible(targets[obj_id], f)]
        if not candidates:
            continue
        frame = max(candidates, key=lambda f: (_margin(targets[obj_id], f), -f))
        by_frame[frame]['obj_ids'] = sorted(set(by_frame[frame]['obj_ids']) | {obj_id})
        covered.add(obj_id)
        changes.append({'kind': 'target_added_to_existing_anchor', 'frame_idx': frame, 'obj_id': obj_id})

    while covered != set(targets):
        remaining = set(targets)-covered
        candidates = []
        for frame in allowed_frames:
            ids = {i for i in remaining if _visible(targets[i], frame)}
            if ids:
                centrality = min(_margin(targets[i], frame) for i in ids)
                candidates.append((len(ids), centrality, -frame, frame, ids))
        if candidates:
            _, _, _, frame, ids = max(candidates, key=lambda item: item[:3])
            origin = 'registered_grid'
        else:
            obj_id = min(remaining)
            span = max(targets[obj_id]['visibility_segments'],
                       key=lambda s: (s['last_frame_idx']-s['first_frame_idx'], -s['first_frame_idx']))
            frame = (span['first_frame_idx']+span['last_frame_idx']) // 2
            ids = {i for i in remaining if _visible(targets[i], frame)}
            origin = 'exact_native_midpoint_no_supported_grid_frame'
            changes.append({'kind': 'exact_native_frame_added_outside_grid', 'frame_idx': frame,
                            'obj_ids': sorted(ids), 'reason': 'No allowed grid candidate lies inside these predicted visibility intervals'})
        if frame in by_frame:
            by_frame[frame]['obj_ids'] = sorted(set(by_frame[frame]['obj_ids']) | ids)
        else:
            by_frame[frame] = {'frame_idx': frame, 'obj_ids': sorted(ids),
                               'reason': 'Recovery initialization supported by saved model visibility; no GT used.'}
        covered |= ids
        changes.append({'kind': 'coverage_anchor_added', 'frame_idx': frame, 'obj_ids': sorted(ids), 'origin': origin})

    anchors = []
    for frame in sorted(by_frame):
        anchor = by_frame[frame]
        if anchor['obj_ids']:
            anchors.append(anchor)
        else:
            changes.append({'kind': 'empty_anchor_removed', 'frame_idx': frame})
    result['anchors'] = anchors
    if len(anchors) > SOFT_ANCHOR_LIMIT:
        changes.append({'kind': 'soft_anchor_limit_exceeded', 'original_limit': SOFT_ANCHOR_LIMIT,
                        'actual_max_anchors': len(anchors), 'reason': 'Preserve every saved target identity; SAM has no three-anchor limit'})
    assert result['targets'] == raw['targets']
    assert {i for a in anchors for i in a['obj_ids']} == set(targets)
    assert all(_visible(targets[i], a['frame_idx']) for a in anchors for i in a['obj_ids'])
    return result, changes


def canonicalize_boxes(expectedids, rawboxes):
    """Return (usable boxes, missing usable IDs, ambiguous IDs, changes).

    Missing IDs include ambiguous IDs, because neither has a usable box. Exact
    coordinate duplicates are safe to collapse; different coordinates for the
    same ID are never selected by confidence or by GT.
    """
    if (not isinstance(expectedids, (list, tuple)) or not expectedids
            or any(type(i) is not int or i < 0 for i in expectedids)
            or len(set(expectedids)) != len(expectedids)):
        raise ValueError('expectedids must contain unique nonnegative integer IDs')
    if isinstance(rawboxes, dict):
        rawboxes = rawboxes.get('boxes')
    if not isinstance(rawboxes, list):
        raise ValueError('rawboxes must be a list or a boxes response dictionary')
    groups = {i: {} for i in expectedids}
    changes = []
    for index, box in enumerate(rawboxes):
        if not isinstance(box, dict) or type(box.get('obj_id')) is not int:
            changes.append({'kind': 'invalid_box_ignored', 'source_box_index': index, 'reason': 'invalid object ID'})
            continue
        obj_id = box['obj_id']
        if obj_id not in groups:
            changes.append({'kind': 'unexpected_box_ignored', 'source_box_index': index, 'obj_id': obj_id})
            continue
        coords = box.get('box_2d')
        confidence = box.get('confidence')
        valid = (isinstance(coords, list) and len(coords) == 4
                 and all(type(v) is int and 0 <= v <= 1000 for v in coords)
                 and coords[0] < coords[2] and coords[1] < coords[3]
                 and type(confidence) in (int, float) and math.isfinite(confidence)
                 and 0 <= confidence <= 1 and isinstance(box.get('reason'), str))
        if not valid:
            changes.append({'kind': 'invalid_box_ignored', 'source_box_index': index,
                            'obj_id': obj_id, 'reason': 'invalid bounds, area, confidence, or reason'})
            continue
        key = tuple(coords)
        if key in groups[obj_id]:
            changes.append({'kind': 'identical_box_duplicate_removed', 'source_box_index': index, 'obj_id': obj_id})
        else:
            groups[obj_id][key] = deepcopy(box)
    usable = []
    ambiguous = []
    for obj_id in expectedids:
        variants = groups[obj_id]
        if len(variants) == 1:
            usable.append(next(iter(variants.values())))
        elif len(variants) > 1:
            ambiguous.append(obj_id)
            changes.append({'kind': 'different_boxes_need_grounding', 'obj_id': obj_id,
                            'box_variants': [list(coords) for coords in variants]})
    usable_ids = {box['obj_id'] for box in usable}
    missing = [i for i in expectedids if i not in usable_ids]
    return usable, missing, ambiguous, changes
