"""Retained GT-free historical recovery function sequence (transport supplied by caller)."""
from __future__ import annotations
import copy,json,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
from .headless_worker import load_source,part_ledger
from .recovery_calls import RecoveryCalls,FatalAPIError,UnresolvedCallError
from pipeline.svag_dense_sampling import build_tracking_map
from pipeline.svag_repair import repair_anchor_selection,canonicalize_boxes,NeedsTemporalRepair
MODEL='gemini-3.7-flash'
ACCOUNT=None
PROJECT=None

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()

class CapturedRequest(Exception):
    pass

def request_parts(module, callback):
    captured = []
    original = module.base.call_function
    def capture(client, model, name, schema, parts):
        captured.append((name, schema, parts))
        raise CapturedRequest()
    module.base.call_function = capture
    try:
        callback()
    except CapturedRequest:
        pass
    finally:
        module.base.call_function = original
    if len(captured) != 1:
        raise ValueError('Expected exactly one frozen planner request')
    return captured[0]

def roster(selection):
    return sorted((t['obj_id'], t['object_label'], t['visual_identity_description'])
                  for t in selection['targets'])

def original_records(source, sample):
    directory = source/'svag_v3/plans'/sample
    return [(p, read(p)) for p in sorted((directory/'calls').glob('call_*.json'))]

def raw_for(records, name):
    matches = [(p, v) for p, v in records if v.get('function_name') == name and v.get('raw_arguments')]
    return matches[-1][1]['raw_arguments'] if matches else None

def prior_boxes(records):
    by_frame = {}
    for path, value in records:
        if value.get('function_name') != 'submit_multi_anchor_boxes' or not value.get('raw_arguments'):
            continue
        texts = ' '.join(p.get('text', '') for p in value.get('input_parts', []))
        match = re.search(r'Exact original frame (\d+)', texts)
        if match:
            by_frame[int(match.group(1))] = value['raw_arguments'].get('boxes', [])
    return by_frame

def recover_sample(source, output, manifest, query, client):
    sample = query['sample']; directory = output/'plans'/sample
    receipt_path = directory/'recovery_receipt.json'; plan_path = directory/'multi_anchor_plan.json'
    if receipt_path.is_file() and plan_path.is_file():
        receipt = read(receipt_path)
        if receipt.get('status') == 'complete' and sha(plan_path) == receipt['plan_sha256']:
            return receipt
    directory.mkdir(parents=True, exist_ok=True)
    original = original_records(source, sample)
    arm = manifest['arms']['svag_v3']
    module = load_source(arm['module_path'], arm['base_dir'], 'svag_v3')
    proxy = Path(query['proxy_path']); video = Path(query['video_path'])
    for path, expected in [(proxy, query['proxy_sha256']), (video, query['source_video_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'Original media changed: {path}')
    calls = RecoveryCalls(client, directory/'calls', MODEL, ACCOUNT, PROJECT, max_attempts=3)
    changes = []; noun = raw_for(original, 'submit_answer_noun_phrase')
    verification = raw_for(original, 'submit_answer_verification')
    old_selection = raw_for(original, 'submit_multi_anchor_selection')
    fresh = old_selection is None or sample == 'video_7_q923'
    if fresh:
        request = request_parts(module, lambda: module.request_noun_phrase(
            None, MODEL, proxy, query['question'], manifest['video_fps'], planning_task='svag'))
        noun, metadata = calls.invoke(*request, module.base.NounPhraseResponse.model_validate,
            tag='fresh_noun', feedback=(
                'The earlier run did not produce a usable grounding. Reinspect the actual video; '
                'do not copy unsupported earlier conclusions. Original-frame labels are authoritative.'))
        noun = noun.model_dump(mode='json')
        changes.append({'kind': 'fresh_noun_recovery', 'metadata': metadata})
        verification = None
    if noun is None:
        raise ValueError('No answer noun phrase available')
    phrase = noun['answer_noun_phrase']; noun_reason = noun['reasoning']
    if verification:
        corrected = (verification.get('corrected_noun_phrase') or '').strip()
        if verification.get('verdict') == 'wrong' and corrected and corrected.lower() != 'none':
            phrase = corrected
    locked = None if fresh else roster(old_selection)
    locked_targets = None if fresh else copy.deepcopy(old_selection['targets'])

    def get_selection(feedback, tag):
        maximum = max(3, len(old_selection.get('anchors', [])) if old_selection else 3)
        request = request_parts(module, lambda: module.request_anchor_selection(None, MODEL,
            proxy, query['question'], phrase, query['allowed_anchor_frames'], maximum,
            manifest['video_fps'], planning_task='svag', num_frames=query['num_frames']))
        def validate(raw):
            if locked is not None and roster(raw) != locked:
                raise ValueError('Recovery must retain every original target ID, label and description')
            repaired, edits = repair_anchor_selection(raw, query['allowed_anchor_frames'], query['num_frames'])
            selection = module.SVAGAnchorSelectionResponse.model_validate(repaired)
            module.validate_svag_selection(selection, query['num_frames'])
            return selection
        suffix = feedback
        if locked is not None:
            suffix += '\nKeep this complete target roster verbatim; timing and anchors may be corrected:\n'
            suffix += json.dumps(locked_targets, ensure_ascii=False)
        selection, metadata = calls.invoke(*request, validate, tag=tag, feedback=suffix)
        changes.append({'kind': tag, 'metadata': metadata})
        return selection

    if fresh:
        feedback = 'Choose exact original frames with clearly visible targets and supported visual evidence.'
        if sample == 'video_7_q923':
            feedback += (' An earlier frame 283 could not be grounded. Reinspect the whole video '
                         'and choose a supported anchor instead of a zero-area placeholder.')
        selection = get_selection(feedback, 'fresh_selection')
        locked = roster(selection.model_dump(mode='json'))
        locked_targets = [target.model_dump(mode='json') for target in selection.targets]
    else:
        try:
            repaired, edits = repair_anchor_selection(old_selection, query['allowed_anchor_frames'], query['num_frames'])
            changes.extend(edits)
            selection = module.SVAGAnchorSelectionResponse.model_validate(repaired)
            module.validate_svag_selection(selection, query['num_frames'])
        except NeedsTemporalRepair as error:
            selection = get_selection('Repair the contradictory timing in the earlier model response: '
                + str(error) + '. Use actual video evidence; action segments must lie within the '
                'reported visibility segments. Retain all physical target identities.', 'temporal_repair')
    existing_boxes = prior_boxes(original)
    box_metadata = {}; saved = []
    for replan in range(3):
        anchors = selection.anchors
        frames = module.base.load_video_frames_at(video, [a.frame_idx for a in anchors])
        exact = {}
        for anchor in anchors:
            p = directory/'anchor_frames'/f'frame_{anchor.frame_idx:06d}.png'
            module.base.save_rgb(frames[anchor.frame_idx], p); exact[anchor.frame_idx] = p.resolve()
        del frames
        saved = []; identities = {t.obj_id: t for t in selection.targets}
        failed = None
        for anchor in anchors:
            usable, missing, ambiguous, edits = canonicalize_boxes(anchor.obj_ids,
                existing_boxes.get(anchor.frame_idx, []))
            changes.extend({'frame_idx': anchor.frame_idx, **e} for e in edits)
            if missing:
                requested_anchor = anchor.model_copy(update={'obj_ids': missing})
                request = request_parts(module, lambda: module.request_anchor_boxes(None, MODEL,
                    proxy, exact[anchor.frame_idx], query['question'], phrase, requested_anchor,
                    [identities[i] for i in missing], manifest['video_fps']))
                def validate(raw, expected=missing):
                    result = module.AnchorBoxesResponse.model_validate(raw)
                    if sorted(b.obj_id for b in result.boxes) != sorted(expected):
                        raise ValueError(f'Return exactly one positive box for each requested ID {expected}')
                    return result
                try:
                    result, metadata = calls.invoke(*request, validate,
                        tag=f'bbox_{anchor.frame_idx}_'+','.join(map(str, missing)),
                        feedback='Repair only the missing or ambiguous boxes. Inspect the exact original '
                            'image. Keep IDs and physical identities; do not invent boxes for absent '
                            'actors or use a zero-area placeholder. Return each requested ID once.')
                except (FatalAPIError, UnresolvedCallError):
                    raise
                except Exception as error:
                    failed = f'Exact frame {anchor.frame_idx}, IDs {missing}: {error}'
                    break
                usable.extend(b.model_dump(mode='json') for b in result.boxes)
                box_metadata[str(anchor.frame_idx)] = metadata
            else:
                box_metadata[str(anchor.frame_idx)] = {'source': 'unchanged original valid boxes',
                    'new_model_calls': 0}
            boxes = sorted(usable, key=lambda b: b['obj_id'])
            if sorted(b['obj_id'] for b in boxes) != sorted(anchor.obj_ids):
                raise ValueError('Final anchor boxes do not match the requested identities')
            saved.append(module.SavedAnchor(frame_idx=anchor.frame_idx, sam_frame_idx=0,
                obj_ids=anchor.obj_ids, reason=anchor.reason,
                boxes=[module.AnchorBox.model_validate(b) for b in boxes],
                exact_frame_path=str(exact[anchor.frame_idx]), call_metadata=box_metadata[str(anchor.frame_idx)]))
        if failed is None:
            break
        if replan == 2:
            raise ValueError(failed)
        selection = get_selection('Some exact-frame boxes could not be grounded: '+failed+
            '. Correct the anchor and timing using the actual labeled video. Keep every target identity. '
            'Choose frames where the requested objects are visible.', f'anchor_reground_{replan+1}')
    anchor_cap = max(manifest['max_anchor_frames'], len(saved))
    plan = module.SavedMultiAnchorPlan(video_id=query['video_id'], question_id=str(query['question_id']),
        question=query['question'], answer_noun_phrase=phrase, noun_reasoning=noun_reason,
        noun_confidence=noun['confidence'], decision_summary=selection.decision_summary,
        first_visible_frame_idx=selection.first_visible_frame_idx,
        last_visible_frame_idx=selection.last_visible_frame_idx,
        visibility_boundary_reason=selection.visibility_boundary_reason,
        targets=selection.targets, anchors=saved,
        sam_stride=1 if query['source_fps']<=10 else max(1, round(query['source_fps']/10)),
        sam_target_fps=min(float(query['source_fps']), 10.), required_frame_stride=0,
        boundary_radius=__import__('math').ceil(query['source_fps']*.5),
        max_anchor_frames=anchor_cap, anchor_policy='gemini_failed_input_repair', model=MODEL,
        gemini_video_fps=manifest['video_fps'], planning_task='svag', planner_revision='svag-v3',
        planner_provider='gemini', planner_input_format='labeled_video_proxy', planner_frame_limit=192,
        call_metadata={'recovery': {'revision': 'svag-v3-required-input-repair-v1',
            'source_experiment': str(source), 'GT_in_model_inputs': False,
            'account': ACCOUNT, 'project': PROJECT, 'changes': changes,
            'source_call_sha256': {str(p): sha(p) for p, _ in original},
            'exact_frame_sha256': {str(p): sha(p) for p in exact.values()},
            'original_target_roster_preserved': not fresh,
            'fresh_recovery_roster_locked_for_subsequent_calls': fresh,
            'anchor_cap_exception': anchor_cap > manifest['max_anchor_frames']}})
    mapping = build_tracking_map(plan.model_dump(mode='json'), query['num_frames'], query['source_fps'])
    for anchor in plan.anchors:
        anchor.sam_frame_idx = mapping['anchor_sam_indices'][str(anchor.frame_idx)]
    save(plan_path, plan.model_dump(mode='json'))
    map_path = directory/'sam_frame_map.json'; save(map_path, mapping)
    save(directory/'anchor_selection.json', selection.model_dump(mode='json'))
    receipt = {'sample': sample, 'status': 'complete', 'finished_at': now(),
        'plan_path': str(plan_path.resolve()), 'plan_sha256': sha(plan_path),
        'sam_frame_map_path': str(map_path.resolve()), 'recovery_receipt_path': str(receipt_path.resolve()),
        'targets': len(plan.targets), 'anchors': len(plan.anchors), 'changes': changes,
        'GT_used': False, 'source_raw_sha256': {str(p): sha(p) for p, _ in original}}
    save(receipt_path, receipt)
    return receipt
