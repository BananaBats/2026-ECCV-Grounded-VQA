#!/usr/bin/env python3
"""Headless paired Perception planner worker using immutable source modules.

The original --plan-only option exits before boxes. This adapter invokes the
original planning functions through exact-anchor grounding, then stops before
SAM. Prompts, schemas, critic behavior and model outputs remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('original', 'svag_v3')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)


def classify_error(error, last_record=None):
    message = str(error)
    code = getattr(error, 'code', None)
    if not isinstance(code, int):
        match = re.search(r'\b(4\d\d|5\d\d)\s+(?:[A-Z_]{3,}|HTTP)', message)
        code = int(match.group(1)) if match else None
    if code in {401, 403} or any(token in message for token in (
            'DefaultCredentialsError', 'RefreshError', 'UNAUTHENTICATED', 'PERMISSION_DENIED')):
        return 'api_access', code
    if any(token in message.lower() for token in ('source changed', 'manifest changed', 'media/source changed',
                                                  'anchor image changed', 'source mismatch')):
        return 'source_integrity', code
    if last_record and last_record.get('status') == 'validation_failed':
        return 'model_output_validation', code
    if code is not None or (last_record and last_record.get('status') == 'call_failed'):
        return 'api_error', code
    return 'configuration', code


def resolve_path(value, experiment):
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = experiment/path
    return candidate if candidate.exists() else ROOT/path


def load_source(module_path, base_dir, arm):
    """Each CLI invocation handles one arm; also avoid stale modules in tests."""
    module_path, base_dir = Path(module_path).resolve(), Path(base_dir).resolve()
    base_path = base_dir/'box_baseline.py'
    if not module_path.is_file() or not base_path.is_file():
        raise FileNotFoundError('Pinned planner/base source is missing')
    os.environ['V6_ROOT'] = str(base_dir)
    sys.path.insert(0, str(base_dir))
    previous = sys.modules.get('box_baseline')
    if previous is not None and Path(previous.__file__).resolve() != base_path:
        del sys.modules['box_baseline']
    key = f'_perception_pinned_{arm}_{digest(module_path)[:12]}'
    spec = importlib.util.spec_from_file_location(key, module_path)
    if spec is None or spec.loader is None:
        raise ValueError('Could not import pinned planner source')
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    if Path(module.base.__file__).resolve() != base_path:
        raise ValueError('Pinned planner imported a different box_baseline module')
    return module


def part_ledger(part):
    result = {}
    if getattr(part, 'text', None) is not None:
        result['text'] = part.text
    inline = getattr(part, 'inline_data', None)
    if inline is not None:
        data = inline.data or b''
        if not isinstance(data, bytes):
            raise TypeError('Inline input must contain bytes, not encoded text')
        result.update(mime_type=inline.mime_type, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    video = getattr(part, 'video_metadata', None)
    if video is not None:
        result['video_metadata'] = video.model_dump(mode='json', exclude_none=True)
    file_data = getattr(part, 'file_data', None)
    if file_data is not None:
        result['file_data'] = file_data.model_dump(mode='json', exclude_none=True)
    return result


class Journal:
    def __init__(self, module, directory, guard):
        self.module, self.directory, self.guard = module, Path(directory), guard
        self.paths = []
        self.original = module.base.call_function

    def call(self, client, model, name, schema, parts):
        self.guard()
        path = self.directory/f'call_{len(self.paths)+1:02d}_{name}.json'
        if path.exists():
            raise ValueError('Refusing to overwrite an existing model journal')
        record = {'status': 'calling', 'started_at': now(), 'model': model, 'function_name': name,
                  'schema_sha256': hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                  'input_parts': [part_ledger(part) for part in parts], 'semantic_retries': 0}
        save(path, record)
        self.paths.append(path)
        try:
            arguments, metadata = self.original(client, model, name, schema, parts)
            record.update(status='returned', raw_arguments=arguments, call_metadata=metadata, returned_at=now())
            save(path, record)
            return arguments, metadata
        except Exception as error:
            record.update(status='call_failed', error_type=type(error).__name__, error=str(error), finished_at=now())
            save(path, record)
            raise

    def invoke(self, callback):
        before = len(self.paths)
        try:
            value = callback()
        except Exception as error:
            if len(self.paths) > before:
                path = self.paths[-1]
                record = json.loads(path.read_text())
                if record.get('status') == 'returned':
                    record.update(status='validation_failed', error_type=type(error).__name__,
                                  error=str(error), finished_at=now())
                    save(path, record)
            raise
        if len(self.paths) > before:
            path = self.paths[-1]
            record = json.loads(path.read_text())
            if record.get('status') == 'returned':
                record.update(status='complete', finished_at=now())
                save(path, record)
        return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--project', default=os.environ.get('GOOGLE_CLOUD_PROJECT'))
    parser.add_argument('--location', default=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def run_worker(args, *, client=None):
    started = time.monotonic()
    experiment = Path(args.experiment_root).resolve()
    manifest_path = experiment/'inference_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = digest(manifest_path)
    rows = manifest['queries']
    samples = [row['sample'] for row in rows]
    if len(samples) != len(set(samples)):
        raise ValueError('Inference manifest contains duplicate sample IDs')
    matches = [row for row in rows if row['sample'] == args.sample]
    if len(matches) != 1 or Path(args.sample).name != args.sample:
        raise ValueError('Requested sample is not a unique safe inference-manifest key')
    row = matches[0]
    video_id, question_id = str(row['video_id']), str(row['question_id'])
    if args.sample != f'{video_id}_q{question_id}' or not isinstance(row['question'], str) or not row['question'].strip():
        raise ValueError('Sample/video/question identity is inconsistent')
    question = row['question']
    if any(key.lower() in {'gt', 'ground_truth', 'answer', 'answers', 'reference_answer', 'answer_id', 'answer_track_ids',
                           'spatial', 'tracks', 'track_ids', 'action_intervals', 'object_tracking',
                           'bounding_boxes', 'initial_tracking_box', 'gt_tracks'} for key in row):
        raise ValueError('Inference query record must not include ground truth or reference answers')
    source = manifest['arms'][args.arm]
    module_path = resolve_path(source['module_path'], experiment)
    base_dir = resolve_path(source.get('base_dir', str(module_path.parent)), experiment)
    video_path = resolve_path(row['video_path'], experiment)
    proxy_path = resolve_path(row['proxy_path'], experiment)
    proxy_metadata_path = resolve_path(row['proxy_metadata_path'], experiment)
    proxy_metadata = json.loads(proxy_metadata_path.read_text())
    num_frames = int(row['num_frames'])
    allowed = list(row['allowed_anchor_frames'])
    if (not allowed or len(set(allowed)) != len(allowed) or
            any(type(frame) is not int or not 0 <= frame < num_frames for frame in allowed)):
        raise ValueError('Anchor candidates must be unique valid original-frame IDs')
    if proxy_metadata.get('original_frame_count', num_frames) != num_frames:
        raise ValueError('Planning proxy and source video frame counts differ')
    model, fps = manifest['model'], float(manifest.get('video_fps', 4.0))
    max_anchors = int(manifest.get('max_anchor_frames', 3))
    stride = int(manifest.get('anchor_frame_stride', manifest.get('sam_stride', 30)))
    boundary_radius = int(manifest.get('boundary_radius', 15))
    if max_anchors < 1 or stride < 1 or not 0 < fps <= 24:
        raise ValueError('Invalid registered planner/video settings')
    input_paths = [video_path, proxy_path, proxy_metadata_path, module_path, base_dir/'box_baseline.py']
    input_hashes = {str(path.resolve()): digest(path) for path in input_paths}
    for path, key in ((video_path, 'source_video_sha256'), (proxy_path, 'proxy_sha256')):
        expected = row.get(key)
        if expected is not None and input_hashes[str(path.resolve())] != expected:
            raise RuntimeError(f'Registered inference media/source changed: {key}')
    registered = manifest.get('source_sha256', {})

    def guard():
        if digest(manifest_path) != manifest_sha:
            raise RuntimeError('Inference manifest changed during this worker')
        for name, expected in registered.items():
            if digest(resolve_path(name, experiment)) != expected:
                raise RuntimeError(f'Registered inference source changed: {name}')
        for name, expected in input_hashes.items():
            if digest(name) != expected:
                raise RuntimeError(f'Inference media/source changed: {name}')

    guard()
    directory = experiment/args.arm/'plans'/args.sample
    plan_path = directory/'multi_anchor_plan.json'
    module = load_source(module_path, base_dir, args.arm)
    if plan_path.exists():
        plan = module.SavedMultiAnchorPlan.model_validate_json(plan_path.read_text())
        provenance = plan.call_metadata.get('perception_comparison', {})
        if ((plan.video_id, plan.question_id, plan.question, plan.model) != (video_id, question_id, question, model)
                or provenance.get('arm') != args.arm or provenance.get('inference_manifest_sha256') != manifest_sha):
            raise ValueError('Existing plan configuration/source mismatch')
        for path, sha in provenance.get('exact_frame_sha256', {}).items():
            if digest(path) != sha:
                raise RuntimeError('Saved exact anchor image changed')
        result = {'status': 'complete', 'sample': args.sample, 'arm': args.arm, 'resumed': True,
                  'plan_path': str(plan_path), 'targets': len(plan.targets), 'anchors': len(plan.anchors),
                  'failure_class': None, 'error': None, 'http_code': None, 'seconds': time.monotonic()-started}
        save(directory/'worker_result.json', result)
        return result
    if (directory/'calls').exists() and any((directory/'calls').iterdir()):
        raise ValueError('Partial model journals exist; no automatic semantic retry or overwrite is allowed')
    if args.dry_run:
        return {'status': 'dry_run', 'sample': args.sample, 'arm': args.arm, 'model': model,
                'question': question, 'allowed_anchor_frames': allowed, 'max_anchor_frames': max_anchors,
                'proxy_sha256': input_hashes[str(proxy_path.resolve())], 'module_path': str(module_path)}
    if client is None:
        if not args.project:
            raise ValueError('A Google Cloud project is required for model inference')
        from google import genai
        from google.genai import types
        client = genai.Client(vertexai=True, project=args.project, location=args.location,
                              http_options=types.HttpOptions(api_version='v1', timeout=300_000))
    directory.mkdir(parents=True, exist_ok=True)
    journal = Journal(module, directory/'calls', guard)
    module.base.call_function = journal.call
    result = {'sample': args.sample, 'arm': args.arm, 'started_at': now(), 'status': 'planning',
              'GT_in_model_inputs': False, 'SAM_started': False}
    save(directory/'run_result.json', result)
    kwargs = {'planning_task': 'svag'} if args.arm == 'svag_v3' else {}
    try:
        noun, noun_metadata = journal.invoke(lambda: module.request_noun_phrase(
            client, model, proxy_path, question, fps, **kwargs))
        phrase, reasoning = noun.answer_noun_phrase, noun.reasoning
        verification, verification_metadata = journal.invoke(lambda: module.verify_noun_phrase(
            client, model, proxy_path, question, phrase, fps, **kwargs))
        corrected = (verification.get('corrected_noun_phrase') or '').strip()
        if verification.get('verdict') == 'wrong' and corrected:
            reasoning = f"[critic override of '{phrase}'] "+str(verification.get('reasoning', ''))
            phrase = corrected
        count = verification.get('expected_instance_count')
        hint = count if isinstance(count, int) and count >= (1 if args.arm == 'svag_v3' else 2) else None
        noun_metadata = {**noun_metadata, 'noun_verification': {**verification, 'call_metadata': verification_metadata}}

        def select():
            extra = {**kwargs, 'num_frames': num_frames} if args.arm == 'svag_v3' else {}
            selection, metadata = module.request_anchor_selection(
                client, model, proxy_path, question, phrase, allowed, max_anchors, fps,
                instance_count_hint=hint, **extra)
            if args.arm == 'svag_v3':
                module.validate_svag_selection(selection, num_frames)
            return selection, metadata

        selection, selection_metadata = journal.invoke(select)
        save(directory/'anchor_selection.json', selection.model_dump(mode='json'))
        sampled, sampling = module.segment_boundary_dense_frames(
            num_frames, stride, selection, boundary_radius)
        identities = {target.obj_id: target for target in selection.targets}
        saved_anchors = []
        exact_hashes = {}
        for anchor in selection.anchors:
            frame = module.base.load_video_frames_at(video_path, [anchor.frame_idx])[anchor.frame_idx]
            exact_path = module.base.save_rgb(frame, directory/'anchor_frames'/f'frame_{anchor.frame_idx:06d}.png')
            exact_path = Path(exact_path).resolve()
            exact_hashes[str(exact_path)] = digest(exact_path)
            boxes, metadata = journal.invoke(lambda: module.request_anchor_boxes(
                client, model, proxy_path, exact_path, question, phrase, anchor,
                [identities[oid] for oid in anchor.obj_ids], fps))
            saved_anchors.append(module.SavedAnchor(
                frame_idx=anchor.frame_idx, sam_frame_idx=sampled.index(anchor.frame_idx), obj_ids=anchor.obj_ids,
                reason=anchor.reason, boxes=boxes.boxes, exact_frame_path=str(exact_path), call_metadata=metadata))
        guard()
        if any(digest(path) != sha for path, sha in exact_hashes.items()):
            raise RuntimeError('An exact anchor image changed before plan publication')
        provenance = {'arm': args.arm, 'inference_manifest_sha256': manifest_sha,
                      'planner_module_sha256': input_hashes[str(module_path.resolve())],
                      'source_video_sha256': input_hashes[str(video_path.resolve())],
                      'proxy_sha256': input_hashes[str(proxy_path.resolve())],
                      'anchor_grid': allowed, 'GT_in_model_inputs': False,
                      'SAM_started': False, 'semantic_retries': 0,
                      'headless_adapter': 'original_function_calls_through_bbox_then_stop',
                      'logical_model_calls': len(journal.paths), 'exact_frame_sha256': exact_hashes}
        values = dict(video_id=video_id, question_id=question_id, question=question,
                      answer_noun_phrase=phrase, noun_reasoning=reasoning, noun_confidence=noun.confidence,
                      decision_summary=selection.decision_summary,
                      first_visible_frame_idx=selection.first_visible_frame_idx,
                      last_visible_frame_idx=selection.last_visible_frame_idx,
                      visibility_boundary_reason=selection.visibility_boundary_reason,
                      targets=selection.targets, anchors=saved_anchors, sam_stride=stride,
                      boundary_radius=boundary_radius, max_anchor_frames=max_anchors, anchor_policy='gemini',
                      model=model, gemini_video_fps=fps,
                      call_metadata={'noun_phrase': noun_metadata, 'anchor_selection': selection_metadata,
                                     'gemini_planning_proxy': proxy_metadata, 'perception_comparison': provenance})
        if args.arm == 'svag_v3':
            values.update(planning_task='svag', planner_revision='svag-v3', planner_provider='gemini',
                          planner_input_format='labeled_video_proxy', planner_frame_limit=192)
        plan = module.SavedMultiAnchorPlan(**values)
        save(plan_path, plan.model_dump(mode='json'))
        save(directory/'sam_frame_map.json', {**sampling, 'num_original_frames': num_frames,
             'source_fps': row['source_fps'], 'sam_to_original_frame': {str(i): f for i, f in enumerate(sampled)}})
        result.update(status='complete', finished_at=now(), plan_path=str(plan_path),
                      targets=len(plan.targets), anchors=len(plan.anchors), logical_model_calls=len(journal.paths),
                      failure_class=None, error=None, http_code=None, seconds=time.monotonic()-started)
        save(directory/'run_result.json', result)
        save(directory/'worker_result.json', result)
        return result
    except Exception as error:
        last_record = json.loads(journal.paths[-1].read_text()) if journal.paths else None
        failure_class, code = classify_error(error, last_record)
        result.update(status='failed', finished_at=now(), error_type=type(error).__name__, error=str(error),
                      logical_model_calls=len(journal.paths),
                      last_call=str(journal.paths[-1]) if journal.paths else None,
                      failure_class=failure_class, http_code=code, plan_path=None, targets=None, anchors=None,
                      seconds=time.monotonic()-started)
        save(directory/'run_result.json', result)
        save(directory/'worker_result.json', result)
        raise
    finally:
        module.base.call_function = journal.original


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic()
    try:
        result = run_worker(args)
    except Exception as error:
        directory = Path(args.experiment_root).resolve()/args.arm/'plans'/args.sample
        result_path = directory/'worker_result.json'
        if Path(args.sample).name != args.sample:
            raise
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('status') == 'complete':
                failure_class, code = classify_error(error)
                result = {'sample': args.sample, 'arm': args.arm, 'status': 'failed',
                          'failure_class': failure_class, 'error': str(error), 'http_code': code,
                          'plan_path': None, 'targets': None, 'anchors': None, 'seconds': time.monotonic()-started}
        else:
            failure_class, code = classify_error(error)
            result = {'sample': args.sample, 'arm': args.arm, 'status': 'failed',
                      'failure_class': failure_class, 'error': str(error), 'http_code': code,
                      'plan_path': None, 'targets': None, 'anchors': None, 'seconds': time.monotonic()-started}
        save(result_path, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
