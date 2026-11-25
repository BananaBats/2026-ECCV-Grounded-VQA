#!/usr/bin/env python3
"""Resume a fresh Gemini-only, every-source-frame Perception baseline.

Inference reads only a registered question manifest, JPEGs, and accepted outputs
from this experiment. Annotation access is confined to evaluate_completed(),
which runs after completed predictions have been exported.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline._legacy.direct_amodal import core

MODEL = 'gemini-3.7-flash'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ApiBlocked(RuntimeError):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class CallLimitReached(RuntimeError):
    pass


class ChunkTooLong(RuntimeError):
    pass


class JournalBlocked(RuntimeError):
    pass


def error_code(error: Exception) -> int | None:
    for value in (getattr(error, 'code', None), getattr(error, 'status_code', None)):
        try:
            if value is not None:
                return int(value)
        except (ValueError, TypeError):
            pass
    if isinstance(error, (ValueError, TypeError)):
        return None
    match = re.search(r'\b(401|403|404|429|500|502|503|504)\b', str(error))
    return int(match.group()) if match else None


def verify_manifest(root: Path, manifest: dict[str, Any], *, inference: bool = True) -> None:
    if (manifest.get('method_revision') != core.METHOD_REVISION
            or manifest.get('model') != MODEL or not manifest.get('project')
            or manifest.get('location') != 'global'):
        raise ValueError('Experiment identity, project, model, or method is not registered')
    sources = manifest.get('source_sha256', {})
    required = {str(Path(__file__).resolve()), str((HERE / 'core.py').resolve())}
    if not required.issubset(sources):
        raise ValueError('Frozen source hashes must include runner and core')
    changed = [name for name, value in sources.items() if sha(Path(name)) != value]
    if changed:
        raise ValueError(f'Frozen inference sources changed: {changed}')
    queries = manifest.get('queries', [])
    samples = [query['sample'] for query in queries]
    if not queries or len(samples) != len(set(samples)):
        raise ValueError('Require nonempty, unique registered queries')
    for query in queries:
        if (query['sample'] != f"{query['video_id']}_q{query['question_id']}"
                or Path(query['sample']).name != query['sample']
                or Path(query['video_id']).name != query['video_id']
                or type(query['num_frames']) is not int or query['num_frames'] < 1
                or not query.get('question', '').strip()):
            raise ValueError(f'Invalid registered query: {query.get("sample")}')
        if inference and not Path(query['video_path']).is_file():
            raise FileNotFoundError(query['video_path'])
    sampling = manifest.get('sampling', {})
    if sampling.get('frame_stride') != 1 or sampling.get('context_stride') != 30:
        raise ValueError('Require every source frame and a 30-frame context grid')


def extract_frames(root: Path, query: dict[str, Any]) -> Path:
    """Cache exact decoded frames on disk, sharing the cache across questions."""
    cache_root = root / 'frames'
    directory = cache_root / query['video_id']
    source = Path(query['video_path'])
    stat = source.stat()
    identity = {'video_path': str(source.resolve()), 'source_bytes': stat.st_size,
                'source_mtime_ns': stat.st_mtime_ns, 'num_frames': query['num_frames']}
    manifest_path = directory / 'frame_manifest.json'
    if manifest_path.is_file() and read(manifest_path).get('source') == identity:
        files = sorted(directory.glob('frame_*.jpg'))
        if ([path.name for path in files]
                == [f'frame_{index:06d}.jpg' for index in range(query['num_frames'])]):
            return directory
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{query["video_id"]}.', dir=cache_root))
    try:
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-threads', '1', '-i', str(source), '-map', '0:v:0', '-an',
                   '-threads', '1', '-filter_threads', '1', '-vf', 'settb=1/1,setpts=N', '-r', '1',
                   '-q:v', '2', '-start_number', '0', str(temporary / 'frame_%06d.jpg')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        if result.returncode:
            raise RuntimeError(f'ffmpeg failed: {result.stderr[-4000:]}')
        files = sorted(temporary.glob('frame_*.jpg'))
        expected = [f'frame_{index:06d}.jpg' for index in range(query['num_frames'])]
        if [path.name for path in files] != expected:
            raise ValueError(f'Decoded frame count mismatch: expected {query["num_frames"]}, got {len(files)}')
        save(temporary / 'frame_manifest.json', {
            'source': identity, 'created_at': now(), 'frame_stride': 1,
            'sizes': {str(index): path.stat().st_size for index, path in enumerate(files)},
        })
        if directory.exists():
            directory.rename(cache_root / f'.{query["video_id"]}.replaced_{stamp()}')
        temporary.rename(directory)
        return directory
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _usage(response: Any) -> dict[str, Any] | None:
    value = getattr(response, 'usage_metadata', None)
    if value is None or isinstance(value, dict):
        return value
    return value.model_dump(mode='json')


def _text(response: Any) -> str:
    try:
        return getattr(response, 'text', '') or ''
    except (ValueError, AttributeError):
        return ''


class Runner:
    """Sequential controller; request() is the sole live generation boundary."""
    def __init__(self, root: Path, manifest: dict[str, Any], max_new_calls: int | None = None):
        self.root, self.manifest = Path(root), manifest
        self.max_new_calls, self.new_accepted_calls = max_new_calls, 0
        self.client = self.types = None
        self.completed, self.failed, self.processed = set(), set(), {}
        self._usage_paths = set()
        usage_path = self.root / 'usage.jsonl'
        if usage_path.is_file():
            for line in usage_path.read_text(encoding='utf-8').splitlines():
                try:
                    self._usage_paths.add(json.loads(line).get('attempt_path'))
                except json.JSONDecodeError:
                    pass
        self.current_sample = self.current_frame = self.current_chunk_range = None
        self.current_cap = None
        for query in manifest['queries']:
            sample = query['sample']
            directory = self.root / 'A30' / sample
            result = read(directory / 'run_result.json') if (directory / 'run_result.json').is_file() else {}
            if (result.get('status') == 'complete' and result.get('num_requested_frames') == query['num_frames']
                    and result.get('method_revision') == core.METHOD_REVISION
                    and all((directory / name).is_file() for name in
                            ['predictions.json', 'compatibility_predictions.json', 'naive_predictions.json'])):
                self.completed.add(sample)
                self.processed[sample] = query['num_frames']
            else:
                if result.get('status') == 'failed':
                    self.failed.add(sample)
                progress = read(directory / 'progress.json') if (directory / 'progress.json').is_file() else {}
                self.processed[sample] = min(query['num_frames'], max(0, int(progress.get('num_accepted_frames', 0))))

    def status(self, phase: str = 'running', **extra: Any) -> None:
        total = len(self.manifest['queries'])
        payload = {'phase': phase, 'updated_at': now(), 'pid': os.getpid(),
                   'total': total, 'completed': len(self.completed), 'failed': len(self.failed),
                   'pending': total - len(self.completed) - len(self.failed),
                   'current_sample': self.current_sample, 'current_frame': self.current_frame,
                   'current_chunk_range': self.current_chunk_range,
                   'num_source_frames_processed': sum(self.processed.values()),
                   'total_source_frames': sum(query['num_frames'] for query in self.manifest['queries']),
                   'new_accepted_calls': self.new_accepted_calls,
                   'method_revision': core.METHOD_REVISION, 'model': self.manifest['model']}
        payload.update(extra)
        save(self.root / 'status.json', payload)

    def request(self, prompt: str, image_indices: list[int], frame_dir: Path,
                roles: dict[int, list[str]]) -> Any:
        if self.client is None:
            from google import genai
            from google.genai import types
            self.types = types
            self.client = genai.Client(vertexai=True, project=self.manifest['project'],
                location=self.manifest['location'],
                http_options=types.HttpOptions(api_version='v1', timeout=900000))
        types = self.types
        parts = [types.Part.from_text(text=prompt)]
        for frame in image_indices:
            parts.append(types.Part.from_text(text=f'frame_idx {frame}: roles={",".join(roles[frame])}'))
            parts.append(types.Part.from_bytes(
                data=(frame_dir / f'frame_{frame:06d}.jpg').read_bytes(), mime_type='image/jpeg'))
        generation = self.manifest['generation']
        config = types.GenerateContentConfig(temperature=generation['temperature'],
            thinking_config=types.ThinkingConfig(thinking_level=generation['thinking_level']),
            max_output_tokens=generation['max_output_tokens'], response_mime_type='application/json',
            response_schema=core.NaiveResponse)
        return self.client.models.generate_content(model=self.manifest['model'],
            contents=[types.Content(role='user', parts=parts)], config=config)

    def _log_usage(self, record: dict[str, Any], record_path: Path) -> None:
        if str(record_path) in self._usage_paths:
            return
        usage = {key: record.get(key) for key in ['sample', 'model', 'model_version', 'status',
            'attempt', 'started_at', 'finished_at', 'wall_seconds', 'usage_metadata',
            'http_code', 'error_type', 'requested_frames']}
        usage.update(stage='A30', attempt_path=str(record_path))
        with (self.root / 'usage.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(usage, ensure_ascii=False) + '\n')
        self._usage_paths.add(str(record_path))

    def _attempt_chunk(self, query: dict[str, Any], requested: list[int], context: list[int],
                       handoff: list[dict[str, Any]], registry: dict[str, Any] | None,
                       frame_dir: Path, calls: Path) -> tuple[core.NaiveResponse, dict[str, Any]]:
        video = {'width': query['width'], 'height': query['height'], 'fps': query['source_fps']}
        prompt = core.make_prompt(query['question'], requested, context, video, registry, handoff)
        roles: dict[int, list[str]] = {}
        for name, frames in [('whole_video_context', context),
                             ('previous_handoff', [record['frame_idx'] for record in handoff]),
                             ('requested_output', requested)]:
            for frame in frames:
                roles.setdefault(frame, []).append(name)
        image_indices = sorted(roles)
        validation_failures = transient_failures = attempt = 0
        while True:
            attempt += 1
            base = f'chunk_{requested[0]:06d}_{requested[-1]:06d}.{stamp()}.attempt{attempt:02d}'
            prompt_path, record_path = calls / f'{base}.prompt.txt', calls / f'{base}.json'
            prompt_path.write_text(prompt + '\n', encoding='utf-8')
            record = {'sample': query['sample'], 'method_revision': core.METHOD_REVISION,
                      'model': self.manifest['model'], 'started_at': now(), 'attempt': attempt,
                      'requested_frames': requested, 'context_indices': context,
                      'handoff_frame_indices': [record['frame_idx'] for record in handoff],
                      'input_frame_indices': image_indices, 'prompt_path': str(prompt_path),
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                      'chunk_cap': self.current_cap, 'status': 'requesting'}
            save(record_path, record)
            started, response, caught, parsed = time.monotonic(), None, None, None
            try:
                response = self.request(prompt, image_indices, frame_dir, roles)
                reasons = [str(getattr(c, 'finish_reason', '')) for c in (getattr(response, 'candidates', None) or [])]
                record.update(model_version=getattr(response, 'model_version', None),
                              response_id=getattr(response, 'response_id', None),
                              usage_metadata=_usage(response), finish_reasons=reasons,
                              raw_response_text=_text(response), status='response_received', received_at=now())
                value = getattr(response, 'parsed', None)
                if value is not None:
                    record['parsed_response'] = (value.model_dump(mode='json')
                        if hasattr(value, 'model_dump') else value)
                # Checkpoint the paid response before parsing or committing its
                # accepted chunk. A restart can finish validation without a new call.
                save(record_path, record)
                if any('MAX_TOKENS' in reason.upper() for reason in reasons):
                    raise ChunkTooLong('Model output reached MAX_TOKENS')
                parsed = (core.NaiveResponse.model_validate(value) if value is not None
                          else core.NaiveResponse.model_validate_json(_text(response)))
                core.validate_response(parsed, requested, registry)
                record.update(status='accepted', response=parsed.model_dump(mode='json'))
            except Exception as error:
                caught = error
                code = error_code(error)
                if response is not None:
                    record.setdefault('usage_metadata', _usage(response))
                record.update(status='max_tokens' if isinstance(error, ChunkTooLong) else 'error',
                              error_type=type(error).__name__, error=str(error)[:4000], http_code=code)
            record.update(finished_at=now(), wall_seconds=round(time.monotonic() - started, 3))
            save(record_path, record)
            self._log_usage(record, record_path)
            if caught is None:
                return parsed, record
            if isinstance(caught, ChunkTooLong):
                raise caught
            code = error_code(caught)
            if code in (401, 403, 404):
                raise ApiBlocked(str(caught), code) from caught
            if code in (429, 500, 502, 503, 504) or isinstance(caught, (TimeoutError, ConnectionError)):
                transient_failures += 1
                if transient_failures < 4:
                    delay = 10 * (2 ** (transient_failures - 1))
                    print(f'[{now()}] retry {query["sample"]} API {code} in {delay}s', flush=True)
                    time.sleep(delay)
                    continue
            elif isinstance(caught, (ValueError, TypeError)):
                validation_failures += 1
                if validation_failures < 2:
                    print(f'[{now()}] retry {query["sample"]} validation: {caught}', flush=True)
                    continue
            raise caught

    def _resume(self, query: dict[str, Any], calls: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        frames: list[dict[str, Any]] = []
        registry = None

        def apply_record(record: dict[str, Any], path: Path) -> None:
            nonlocal registry
            requested = record.get('requested_frames', [])
            if (record.get('sample') != query['sample'] or record.get('method_revision') != core.METHOD_REVISION
                    or record.get('model') != self.manifest['model'] or not requested
                    or requested != list(range(len(frames), len(frames) + len(requested)))
                    or requested[-1] >= query['num_frames']):
                raise ValueError(f'Noncontiguous or incompatible accepted chunk: {path.name}')
            parsed = core.NaiveResponse.model_validate(record['response'])
            core.validate_response(parsed, requested, registry)
            if registry is None:
                registry = {'answer': parsed.answer,
                            'targets': [target.model_dump(mode='json') for target in parsed.targets]}
            frames.extend(frame.model_dump(mode='json') for frame in parsed.frames)

        for path in sorted(calls.glob('chunk_*.accepted.json')):
            record = read(path)
            if record.get('status') != 'accepted':
                raise ValueError(f'Invalid accepted chunk state: {path.name}')
            apply_record(record, path)

        # A successful response may have been checkpointed immediately before a
        # process interruption. Promote it offline, preserving the original call.
        while len(frames) < query['num_frames']:
            candidates, unknown = [], []
            for path in sorted(calls.glob(f'chunk_{len(frames):06d}_*.attempt*.json')):
                record = read(path)
                state = record.get('status')
                if state in ('accepted', 'response_received'):
                    reasons = record.get('finish_reasons', [])
                    if any('MAX_TOKENS' in str(reason).upper() for reason in reasons):
                        record['status'] = state = 'max_tokens'
                        save(path, record)
                    else:
                        try:
                            value = record.get('response', record.get('parsed_response'))
                            parsed = (core.NaiveResponse.model_validate(value) if value is not None
                                else core.NaiveResponse.model_validate_json(record.get('raw_response_text', '')))
                            core.validate_response(parsed, record.get('requested_frames', []), registry)
                            record.update(status='accepted', response=parsed.model_dump(mode='json'),
                                          recovered_at=now())
                            candidates.append((path, record))
                        except (ValueError, TypeError) as error:
                            record.update(status='error', error_type=type(error).__name__,
                                          error=f'Offline response recovery: {error}'[:4000])
                            save(path, record)
                            self._log_usage(record, path)
                if state == 'max_tokens':
                    failed_cap = record.get('chunk_cap') or len(record.get('requested_frames', []))
                    if failed_cap <= 12:
                        raise ValueError('Saved response reached MAX_TOKENS at the minimum 12-frame cap')
                    progress_path = calls.parent / 'progress.json'
                    progress = read(progress_path) if progress_path.is_file() else {}
                    cap = min(int(progress.get('adaptive_chunk_frames', self.manifest['sampling']['max_chunk_frames'])),
                              max(12, int(failed_cap) // 2))
                    self._progress(calls.parent, frames, cap)
                    self._log_usage(record, path)
                elif state == 'requesting':
                    unknown.append(path.name)
            if len(candidates) > 1:
                raise JournalBlocked(f'Multiple paid responses need review at frame {len(frames)}')
            if candidates:
                path, record = candidates[0]
                requested = record['requested_frames']
                if (record.get('sample') != query['sample']
                        or record.get('method_revision') != core.METHOD_REVISION
                        or record.get('model') != self.manifest['model'] or not requested
                        or requested != list(range(len(frames), len(frames) + len(requested)))
                        or requested[-1] >= query['num_frames']):
                    raise JournalBlocked(f'Incompatible paid response checkpoint: {path.name}')
                accepted = calls / f'chunk_{requested[0]:06d}_{requested[-1]:06d}.accepted.json'
                save(path, record)
                save(accepted, record)
                apply_record(record, path)
                self._log_usage(record, path)
                print(f'[{now()}] recovered paid response {accepted.name} without another model call', flush=True)
                continue
            if unknown:
                raise JournalBlocked(
                    f'Interrupted request has no response checkpoint; refusing silent rebilling: {unknown}')
            break
        return frames, registry

    def _progress(self, directory: Path, frames: list[dict[str, Any]], cap: int) -> None:
        save(directory / 'progress.json', {'updated_at': now(), 'num_accepted_frames': len(frames),
            'next_frame': len(frames), 'adaptive_chunk_frames': cap})

    def run_sample(self, query: dict[str, Any]) -> dict[str, Any]:
        sample = query['sample']
        if sample in self.completed:
            return {'sample': sample, 'status': 'skipped_complete'}
        directory = self.root / 'A30' / sample
        calls = directory / 'calls'
        calls.mkdir(parents=True, exist_ok=True)
        self.failed.discard(sample)
        self.current_sample, self.current_frame, self.current_chunk_range = sample, 0, None
        self.status()
        started_at = now()
        frames: list[dict[str, Any]] = []
        sampling = self.manifest['sampling']
        cap = int(sampling['max_chunk_frames'])
        try:
            frames, registry = self._resume(query, calls)
            self.processed[sample] = len(frames)
            self.current_frame = len(frames)
            progress_path = directory / 'progress.json'
            if progress_path.is_file():
                old_cap = int(read(progress_path).get('adaptive_chunk_frames', cap))
                if 1 <= old_cap <= cap:
                    cap = old_cap
            if len(frames) < query['num_frames']:
                frame_dir = extract_frames(self.root, query)
                sizes = {index: (frame_dir / f'frame_{index:06d}.jpg').stat().st_size
                         for index in range(query['num_frames'])}
                context = list(range(0, query['num_frames'], sampling['context_stride']))
            while len(frames) < query['num_frames']:
                if (len(frames) < query['num_frames'] and self.max_new_calls is not None
                        and self.new_accepted_calls >= self.max_new_calls):
                    raise CallLimitReached('Accepted model-call limit reached')
                cursor = len(frames)
                handoff = frames[max(0, cursor - sampling['handoff_frames']):cursor]
                prefix = [frame['frame_idx'] for frame in handoff]
                candidates = prefix + list(range(cursor, min(query['num_frames'], cursor + cap)))
                # The temporary prefix makes the first partition include actual
                # accepted handoff images in its byte budget, without resending outputs.
                first = core.partition_frames(candidates, sizes, context,
                    max_frames=cap + len(prefix), max_inline_bytes=sampling['max_inline_bytes'],
                    handoff_frames=sampling['handoff_frames'])[0]
                requested = [frame for frame in first if frame >= cursor]
                if not requested:
                    raise ValueError('No new frame fits with context and accepted handoff')
                self.current_frame, self.current_chunk_range = cursor, [requested[0], requested[-1]]
                self.current_cap = cap
                self.status()
                print(f'[{now()}] {sample} request frames {requested[0]}..{requested[-1]} / {query["num_frames"]}', flush=True)
                try:
                    parsed, record = self._attempt_chunk(query, requested, context, handoff, registry, frame_dir, calls)
                except ChunkTooLong:
                    if cap <= 12:
                        raise ValueError('MAX_TOKENS persisted at the minimum 12-frame cap')
                    cap = max(12, cap // 2)
                    self._progress(directory, frames, cap)
                    print(f'[{now()}] {sample} reduce chunk cap to {cap}; resume at {cursor}', flush=True)
                    continue
                accepted = calls / f'chunk_{requested[0]:06d}_{requested[-1]:06d}.accepted.json'
                if accepted.exists():
                    raise ValueError(f'Accepted chunk already exists: {accepted.name}')
                save(accepted, record)
                if registry is None:
                    registry = {'answer': parsed.answer,
                                'targets': [target.model_dump(mode='json') for target in parsed.targets]}
                frames.extend(frame.model_dump(mode='json') for frame in parsed.frames)
                self.new_accepted_calls += 1
                self.processed[sample], self.current_frame = len(frames), len(frames)
                self._progress(directory, frames, cap)
                self.status()
                print(f'[{now()}] {sample} accepted {len(frames)}/{query["num_frames"]} source frames', flush=True)
                if (len(frames) < query['num_frames'] and self.max_new_calls is not None
                        and self.new_accepted_calls >= self.max_new_calls):
                    raise CallLimitReached('Accepted model-call limit reached')
            direct = core.tracks_from_frames(frames)
            compatible = core.tracks_from_frames(frames, interpolate=True)
            payload = lambda tracks: {query['video_id']: {'grounded_question': {str(query['question_id']): tracks}}}
            save(directory / 'predictions.json', payload(direct))
            save(directory / 'compatibility_predictions.json', payload(compatible))
            save(directory / 'naive_predictions.json', {'video_id': query['video_id'],
                'question_id': query['question_id'], 'method_revision': core.METHOD_REVISION,
                'frame_stride': 1, 'source_fps': query['source_fps'], 'answer': registry['answer'],
                'targets': registry['targets'], 'frames': frames, 'dense_tracks': direct})
            result = {'sample': sample, 'status': 'complete', 'started_at': started_at,
                      'updated_at': now(), 'method_revision': core.METHOD_REVISION,
                      'model': self.manifest['model'], 'num_requested_frames': query['num_frames'],
                      'num_targets': len(registry['targets']), 'num_direct_boxes': sum(len(t['frame_ids']) for t in direct),
                      'num_accepted_chunks': len(list(calls.glob('chunk_*.accepted.json')))}
            save(directory / 'run_result.json', result)
            self.completed.add(sample)
            self.status()
            print(f'[{now()}] {sample} COMPLETE ({len(self.completed)}/{len(self.manifest["queries"])})', flush=True)
            return result
        except (ApiBlocked, CallLimitReached, JournalBlocked, KeyboardInterrupt):
            self.processed[sample] = len(frames)
            self._progress(directory, frames, cap)
            raise
        except Exception as error:
            self.failed.add(sample)
            self.processed[sample] = len(frames)
            self._progress(directory, frames, cap)
            result = {'sample': sample, 'status': 'failed', 'updated_at': now(),
                      'method_revision': core.METHOD_REVISION, 'num_accepted_frames': len(frames),
                      'error_type': type(error).__name__, 'error': str(error)[:4000]}
            save(directory / 'run_result.json', result)
            self.status()
            print(f'[{now()}] {sample} FAILED: {type(error).__name__}: {error}', flush=True)
            return result


def evaluate_completed(root: Path, manifest: dict[str, Any], completed: set[str]) -> dict[str, Any]:
    """GT is used only by evaluator subprocesses, after prediction export."""
    queries = [query for query in manifest['queries'] if query['sample'] in completed]
    if not queries:
        raise ValueError('No completed questions are available for evaluation')
    merged, compatibility = {}, {}
    for query in queries:
        directory = root / 'A30' / query['sample']
        for target, name in [(merged, 'predictions.json'),
                             (compatibility, 'compatibility_predictions.json')]:
            for video_id, video in read(directory / name).items():
                target.setdefault(video_id, {'grounded_question': {}})['grounded_question'].update(video['grounded_question'])
    save(root / 'predictions.json', merged)
    save(root / 'compatibility_predictions.json', compatibility)
    subset = root / 'completed.tsv'
    subset.write_text(''.join(f'{query["video_id"]}\t{query["question_id"]}\n' for query in queries), encoding='utf-8')
    python = HERE.parents[1] / '.venv' / 'bin' / 'python'
    if not python.is_file():
        python = Path(sys.executable)
    for prediction, csv_name in [('predictions.json', 'scores_A30.csv'),
                                 ('compatibility_predictions.json', 'scores_A30_compatibility.csv')]:
        command = [str(python), str(root / 'eval_snapshot' / 'eval_hota.py'), str(root / prediction),
                   '--annotations', manifest['eval_annotations_path'], '--subset', str(subset),
                   '--output-csv', str(root / csv_name)]
        result = subprocess.run(command, capture_output=True, text=True)
        with (root / 'evaluation.log').open('a', encoding='utf-8') as handle:
            handle.write(f'[{now()}] {prediction}\n{result.stdout}\n{result.stderr}\n')
        if result.returncode:
            raise RuntimeError(f'Evaluation failed for {prediction}: {result.stderr[-4000:]}')

    def load_scores(path: Path) -> dict[str, dict[str, str]]:
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len({row['sample'] for row in rows}):
            raise ValueError(f'Duplicate scored sample in {path}')
        return {row['sample']: row for row in rows}

    baseline = load_scores(root / 'baseline_scores_A.csv')
    comparison = {'created_at': now(), 'method_revision': core.METHOD_REVISION,
                  'cohort_total': len(manifest['queries']), 'completed_questions': len(queries),
                  'scope': 'Perception validation; paired completed questions; scores in 0..100 points',
                  'baseline_questions': len(baseline), 'model': manifest['model']}
    for name, csv_name in [('direct', 'scores_A30.csv'), ('compatibility', 'scores_A30_compatibility.csv')]:
        scores = load_scores(root / csv_name)
        matched = sorted(set(scores) & set(baseline))
        if not matched:
            raise ValueError('No paired baseline scores are available')
        old, new = [float(baseline[s]['HOTA']) for s in matched], [float(scores[s]['HOTA']) for s in matched]
        if not all(math.isfinite(value) for value in old + new):
            raise ValueError('Nonfinite HOTA in evaluation output')
        comparison[name] = {'matched_questions': len(matched),
            'baseline_HOTA_points': sum(old) / len(old) * 100,
            'A30_HOTA_points': sum(new) / len(new) * 100,
            'delta_HOTA_points': sum(b - a for a, b in zip(old, new)) / len(old) * 100,
            'improved': sum(b - a > 1e-10 for a, b in zip(old, new)),
            'worsened': sum(b - a < -1e-10 for a, b in zip(old, new)),
            'unchanged': sum(abs(b - a) <= 1e-10 for a, b in zip(old, new))}
    save(root / 'comparison.json', comparison)
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--sample', action='append')
    parser.add_argument('--max-new-calls', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.max_new_calls is not None and args.max_new_calls < 1:
        parser.error('--max-new-calls must be positive')
    if args.dry_run and args.evaluate_only:
        parser.error('--dry-run and --evaluate-only are mutually exclusive')
    root = args.experiment_root.resolve()
    manifest = read(root / 'inference_manifest.json')
    lock = (root / 'run.lock').open('a+', encoding='utf-8')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print('Another controller holds this experiment lock; no duplicate launch.', file=sys.stderr)
        return 2
    lock.seek(0)
    lock.truncate()
    json.dump({'pid': os.getpid(), 'started_at': now()}, lock)
    lock.flush()
    runner = None
    try:
        verify_manifest(root, manifest, inference=not (args.evaluate_only or args.dry_run))
        runner = Runner(root, manifest, args.max_new_calls)
        known = {query['sample'] for query in manifest['queries']}
        selected = set(args.sample) if args.sample else known
        if not selected <= known:
            raise ValueError(f'Unknown sample IDs: {sorted(selected - known)}')
        if args.dry_run:
            runner.status('dry_run', selected_samples=len(selected), API_calls=0)
            print(json.dumps({'status': 'dry_run', 'selected_samples': len(selected),
                              'total': len(known), 'model': manifest['model'], 'API_calls': 0}), flush=True)
            return 0
        runner.status('running', selected_samples=len(selected))
        if not args.evaluate_only:
            for query in manifest['queries']:
                if query['sample'] in selected:
                    runner.run_sample(query)
        all_terminal = len(runner.completed | runner.failed) == len(known)
        if args.evaluate_only or all_terminal:
            runner.current_sample = runner.current_frame = runner.current_chunk_range = None
            runner.status('evaluating')
            comparison = evaluate_completed(root, manifest, runner.completed)
            print(json.dumps(comparison, ensure_ascii=False), flush=True)
        phase = ('complete_with_failures' if runner.failed else 'complete') if all_terminal else 'subset_complete'
        runner.current_sample = runner.current_frame = runner.current_chunk_range = None
        runner.status(phase)
        return 1 if runner.failed else 0
    except CallLimitReached as error:
        runner.status('call_limit_reached', message=str(error))
        print(f'[{now()}] {error}; accepted chunks are resumable.', flush=True)
        return 0
    except ApiBlocked as error:
        runner.status('blocked_api', error_type=type(error).__name__, http_code=error.code, error=str(error)[:4000])
        print(f'[{now()}] STOP API {error.code}: {error}', file=sys.stderr, flush=True)
        return 2
    except JournalBlocked as error:
        runner.status('blocked_journal', error_type=type(error).__name__, error=str(error)[:4000])
        print(f'[{now()}] STOP unresolved call journal: {error}', file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        if runner is not None:
            runner.status('interrupted')
        return 130
    except Exception as error:
        if runner is not None:
            runner.status('failed', error_type=type(error).__name__, error=str(error)[:4000])
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 2
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
