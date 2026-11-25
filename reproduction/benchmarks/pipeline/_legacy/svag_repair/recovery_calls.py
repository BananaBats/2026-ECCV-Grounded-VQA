"""Journal bounded Gemini recovery calls without changing frozen planners.

Requests store prompt text and media hashes, never media bytes or credentials.
A received response is persisted before semantic validation. Interrupted calls
with unknown response status are deliberately not repeated automatically.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from google.genai import types


class FatalAPIError(RuntimeError):
    def __init__(self, message, http_code=None, call_record=None):
        super().__init__(message)
        self.http_code = self.code = http_code
        self.call_record = call_record


class UnresolvedCallError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json', exclude_none=True)
    if hasattr(value, 'value'):
        return _json(value.value)
    return str(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _save(path, payload):
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def _part_ledger(part):
    result = {}
    text = getattr(part, 'text', None)
    if text is not None:
        result['text'] = text
    inline = getattr(part, 'inline_data', None)
    if inline is not None:
        data = inline.data or b''
        if not isinstance(data, bytes):
            raise TypeError('Recovery media parts must contain raw bytes')
        result.update(mime_type=inline.mime_type, bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest())
    video = getattr(part, 'video_metadata', None)
    if video is not None:
        result['video_metadata'] = _json(video)
    file_data = getattr(part, 'file_data', None)
    if file_data is not None:
        # A URI could contain a signed query; retain identity without exposing it.
        result['file_data_sha256'] = _digest(_json(file_data))
    if not result:
        raise ValueError('Unsupported empty recovery input part')
    return result


def _safe_error(error):
    message = str(error)[:8000]
    message = re.sub(r'(?i)Bearer\s+[^\s\"\']+', 'Bearer [redacted]', message)
    message = re.sub(r'(?i)(access_token|refresh_token|client_secret)([\"\']?\s*[:=]\s*[\"\']?)[^\s,\"\']+',
                     r'\1\2[redacted]', message)
    return message


def _code(error):
    for key in ('code', 'status_code'):
        try:
            value = getattr(error, key, None)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _function_config(name, schema):
    # Same settings as the original frozen box_baseline.function_config.
    declaration = types.FunctionDeclaration(name=name,
        description=f'Return the validated {name} result.', parameters_json_schema=schema)
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode='ANY', allowed_function_names=[name])), temperature=0.0)


class RecoveryCalls:
    def __init__(self, client, directory, model, account, project, max_attempts=3):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise ValueError('max_attempts must be an integer in 1..10')
        self.client, self.directory = client, Path(directory)
        self.model, self.account, self.project = model, account, project
        self.max_attempts = max_attempts
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = []

    @staticmethod
    def _metadata(record, path, *, cached):
        return {key: record.get(key) for key in
                ('model', 'function_name', 'usage_metadata', 'request_sha256', 'account', 'project')} | {
                    'call_record': str(path.resolve()), 'cached': cached,
                    'attempt_request_sha256': record.get('attempt_request_sha256')}

    def invoke(self, name, schema, parts, validator, tag, feedback=None):
        """Return validated output and metadata, reusing a successful saved call.

        ``request_sha256`` identifies the original invocation, including caller
        feedback. ``attempt_request_sha256`` additionally hashes retry feedback.
        The overall attempt limit persists across process restarts.
        """
        if not re.fullmatch(r'[A-Za-z_]\w*', name):
            raise ValueError('Invalid requested function name')
        safe_tag = re.sub(r'[^A-Za-z0-9_.-]', '_', str(tag)).strip('._')[:80]
        if not safe_tag:
            raise ValueError('Recovery tag must identify the requested stage')
        initial_parts = list(parts)
        if feedback:
            initial_parts.append(types.Part.from_text(text=str(feedback)))
        initial_ledger = [_part_ledger(part) for part in initial_parts]
        request = {'model': self.model, 'account': self.account, 'project': self.project,
                   'function_name': name, 'schema': schema, 'input_parts': initial_ledger}
        request_sha = _digest(request)
        lock = (self.directory / '.journal.lock').open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._invoke_locked(name, schema, initial_parts, validator, safe_tag,
                                       request, request_sha)
        finally:
            lock.close()

    def _invoke_locked(self, name, schema, initial_parts, validator, tag, request, request_sha):
        saved = []
        all_paths = sorted(self.directory.glob('call_*.json'))
        for path in all_paths:
            record = json.loads(path.read_text())
            if record.get('request_sha256') == request_sha:
                saved.append((path, record))
        for path, record in saved:
            if record['status'] in ('requesting', 'unresolved'):
                raise UnresolvedCallError(f'Unknown response status; inspect {path} before retrying')
        # The process may have stopped after durably receiving a response but
        # before validation/publication. Finish that step without another call.
        for path, record in saved:
            if record['status'] == 'response_received':
                try:
                    names = record.get('response_diagnostics', {}).get('function_call_names', [])
                    if names != [name] or record.get('raw_arguments') is None:
                        raise ValueError(f'Expected one {name} call; got {names}')
                    validated = validator(record['raw_arguments'])
                except Exception as error:
                    record.update(status='validation_failed', error_type=type(error).__name__,
                                  error=_safe_error(error), finished_at=_now())
                    _save(path, record)
                else:
                    record.update(status='complete', finished_at=_now(),
                                  completed_from_received_journal=True)
                    _save(path, record)
                    return validated, self._metadata(record, path, cached=True)
        for path, record in reversed(saved):
            if record['status'] == 'complete':
                validated = validator(record['raw_arguments'])
                return validated, self._metadata(record, path, cached=True)
        if saved and saved[-1][1].get('http_code') in (401, 403, 404):
            previous_path, previous = saved[-1]
            raise FatalAPIError(previous.get('error', 'API access blocked'),
                                previous['http_code'], str(previous_path))
        if len(saved) >= self.max_attempts:
            raise ValueError(f'Recovery attempt limit reached for {tag}: {saved[-1][1].get("error", "invalid response")}')

        previous = saved[-1][1] if saved else None
        serial = max([int(p.name.split('_')[1]) for p in all_paths] or [0])
        for attempt in range(len(saved) + 1, self.max_attempts + 1):
            actual_parts = list(initial_parts)
            if previous and previous.get('status') == 'validation_failed':
                correction = ('The previous response failed this input contract. Correct only the '
                              'reported inconsistency while using the same video, question, and '
                              'requested identities. Do not invent boxes or delete targets to pass validation.\n'
                              + json.dumps({'validation_error': previous.get('error'),
                                            'previous_arguments': previous.get('raw_arguments')},
                                           ensure_ascii=False))[:16000]
                actual_parts.append(types.Part.from_text(text=correction))
            actual_ledger = [_part_ledger(part) for part in actual_parts]
            serial += 1
            path = self.directory / f'call_{serial:04d}_{tag}_attempt{attempt:02d}.json'
            if path.exists():
                raise FileExistsError(f'Recovery journal already exists: {path}')
            record = {**request, 'schema_sha256': _digest(schema),
                      'request_sha256': request_sha,
                      'attempt_request_sha256': _digest({**request, 'input_parts': actual_ledger}),
                      'input_parts': actual_ledger, 'attempt': attempt, 'status': 'requesting',
                      'started_at': _now(), 'response_received': False}
            _save(path, record)
            self.paths.append(path)
            started = time.monotonic()
            try:
                response = self.client.models.generate_content(model=self.model,
                    contents=[types.Content(role='user', parts=actual_parts)],
                    config=_function_config(name, schema))
            except Exception as error:
                code = _code(error)
                record.update(http_code=code, error_type=type(error).__name__, error=_safe_error(error),
                              finished_at=_now(), wall_seconds=time.monotonic()-started,
                              status='api_error' if code is not None else 'unresolved')
                _save(path, record)
                if code in (401, 403, 404):
                    raise FatalAPIError(record['error'], code, str(path)) from error
                if code is None:
                    raise UnresolvedCallError(f'No confirmed response; inspect {path} before retrying') from error
                if code in (408, 429, 500, 502, 503, 504) and attempt < self.max_attempts:
                    previous = record
                    time.sleep(min(20, 2 ** attempt))
                    continue
                raise

            # Persist the raw received payload before semantic checks can fail.
            calls = getattr(response, 'function_calls', None) or []
            matching = len(calls) == 1 and getattr(calls[0], 'name', None) == name
            raw = dict(getattr(calls[0], 'args', None) or {}) if matching else None
            usage = _json(getattr(response, 'usage_metadata', None))
            diagnostics = {'response_id': getattr(response, 'response_id', None),
                           'model_version': getattr(response, 'model_version', None),
                           'prompt_feedback': _json(getattr(response, 'prompt_feedback', None)),
                           'candidates': [{'finish_reason': _json(getattr(c, 'finish_reason', None)),
                                           'safety_ratings': _json(getattr(c, 'safety_ratings', None))}
                                          for c in (getattr(response, 'candidates', None) or [])],
                           'function_call_names': [getattr(c, 'name', None) for c in calls]}
            record.update(status='response_received', response_received=True,
                          received_at=_now(), raw_arguments=raw, usage_metadata=usage,
                          response_diagnostics=diagnostics, wall_seconds=time.monotonic()-started)
            _save(path, record)
            try:
                if not matching:
                    raise ValueError(f'Expected one {name} call; got {diagnostics["function_call_names"]}')
                validated = validator(raw)
            except Exception as error:
                record.update(status='validation_failed', error_type=type(error).__name__,
                              error=_safe_error(error), finished_at=_now())
                _save(path, record)
                previous = record
                if attempt == self.max_attempts:
                    raise
                continue
            record.update(status='complete', finished_at=_now())
            _save(path, record)
            return validated, self._metadata(record, path, cached=False)
        raise RuntimeError('Unreachable recovery call state')
