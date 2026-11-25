#!/usr/bin/env python3
"""Resume VidSTG declarative planning through four durable Gemini Batch waves.

Existing planner functions construct and validate every request. This transport
only replaces inline media with immutable GCS objects and supplies saved batch
function-call results. No dataset GT, SAM inference, or online model calls occur.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

STAGES = ('noun', 'critic', 'selection', 'boxes')
TERMINAL = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_PARTIALLY_SUCCEEDED', 'JOB_STATE_FAILED',
            'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED', 'JOB_STATE_PAUSED'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inline_json_schema(schema):
    """Expand local JSON-schema refs before Batch's BigQuery input import.

    The online API accepts Pydantic's $defs/$ref representation, but Batch
    imports JSON object keys as column names and rejects the dollar signs.
    Keep every constraint and validate responses with the original models.
    """
    def expand(value, stack=()):
        if isinstance(value, list):
            return [expand(item, stack) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        if '$ref' in value:
            ref = value['$ref']
            if not isinstance(ref, str) or not ref.startswith('#/'):
                raise ValueError('Only local JSON-schema references are supported')
            if ref in stack:
                raise ValueError('Recursive JSON schema cannot be inlined for Batch')
            target = schema
            for token in ref[2:].split('/'):
                target = target[token.replace('~1', '/').replace('~0', '~')]
            result = expand(target, (*stack, ref))
        for key, item in value.items():
            if key in {'$ref', '$defs', 'definitions', '$schema'}:
                continue
            if key.startswith('$'):
                raise ValueError('Unsupported Batch JSON-schema keyword: ' + key)
            result[key] = expand(item, stack)
        return result

    return expand(schema)


def request_signature(request):
    """Identify echoed rows without depending on output order or added nulls."""
    contents = []
    for content in request.get('contents', []):
        parts = []
        for part in content.get('parts', []):
            item = {}
            if part.get('text') is not None:
                item['text'] = part['text']
            media = part.get('fileData') or part.get('file_data')
            if media:
                item['fileUri'] = media.get('fileUri', media.get('file_uri'))
            vm = part.get('videoMetadata') or part.get('video_metadata')
            if vm and vm.get('fps') is not None:
                item['fps'] = float(vm['fps'])
            if item:
                parts.append(item)
        contents.append({'role': content.get('role', 'user').lower(), 'parts': parts})
    return digest(json.dumps(contents, ensure_ascii=False, sort_keys=True).encode())


def extract_call(row, expected_name):
    if row.get('status'):
        raise ValueError('Batch row status: ' + str(row['status']))
    response = row.get('response') or {}
    calls = [p['functionCall'] for c in response.get('candidates', [])
             for p in c.get('content', {}).get('parts', []) if p.get('functionCall')]
    if len(calls) != 1 or calls[0].get('name') != expected_name:
        raise ValueError(f'Expected exactly one {expected_name}; calls={[(c.get("name")) for c in calls]}')
    args = calls[0].get('args')
    if not isinstance(args, dict):
        raise ValueError('Function arguments are not an object')
    return args, response


class Cloud:
    def __init__(self, config):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        self.session_class = AuthorizedSession
        self.credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        if self.credentials.quota_project_id != config['project']:
            raise RuntimeError('ADC quota project changed; refusing to use another project')
        self.local = threading.local()
        self.project = config['project']
        self.bucket = config['bucket']

    def request(self, method, url, **kwargs):
        if not hasattr(self.local, 'session'):
            self.local.session = self.session_class(self.credentials)
        idempotent = method == 'GET' or '/upload/storage/' in url
        for attempt in range(3):
            try:
                response = self.local.session.request(method, url, timeout=90, **kwargs)
            except Exception:
                if not idempotent or attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code in {429, 500, 502, 503, 504} and idempotent and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            break
        if not response.ok:
            raise RuntimeError(f'{method} {url.split("?")[0]}: HTTP {response.status_code}: {response.text[:1600]}')
        return response

    def ensure_bucket(self):
        url = 'https://storage.googleapis.com/storage/v1/b/' + self.bucket
        try:
            self.request('GET', url)
            return
        except RuntimeError as exc:
            if 'HTTP 404:' not in str(exc):
                raise
        self.request('POST', 'https://storage.googleapis.com/storage/v1/b',
                     params={'project': self.project}, json={
                         'name': self.bucket, 'location': 'US',
                         'iamConfiguration': {'uniformBucketLevelAccess': {'enabled': True},
                                              'publicAccessPrevention': 'enforced'},
                         'softDeletePolicy': {'retentionDurationSeconds': '0'}})

    def upload(self, name, data, mime):
        url = f'https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o'
        try:
            self.request('POST', url, params={'uploadType': 'media', 'name': name, 'ifGenerationMatch': '0'},
                         data=data, headers={'Content-Type': mime})
        except RuntimeError as exc:
            if 'HTTP 412:' not in str(exc):
                raise
            meta = self.request('GET', f'https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{quote(name, safe="")}').json()
            md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
            if int(meta['size']) != len(data) or meta.get('md5Hash') != md5:
                raise RuntimeError('Immutable GCS input changed: ' + name)
        return f'gs://{self.bucket}/{name}'

    def list_output(self, prefix):
        items, token = [], None
        while True:
            params = {'prefix': prefix}
            if token:
                params['pageToken'] = token
            obj = self.request('GET', f'https://storage.googleapis.com/storage/v1/b/{self.bucket}/o', params=params).json()
            items.extend(obj.get('items', []))
            token = obj.get('nextPageToken')
            if not token:
                return items

    def download(self, name):
        return self.request('GET', f'https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{quote(name, safe="")}', params={'alt': 'media'}).content


class CapturedRequest(Exception):
    def __init__(self, name, schema, parts):
        self.name, self.schema, self.parts = name, schema, parts


class Planner:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['release_root'])
        self.run = Path(config['run_root'])
        os.environ['VQA_PROJECT_ROOT'] = str(self.root)
        os.environ['VQA_DATA_SPLIT'] = 'vidstg_test'
        os.environ['VQA_VALID_ROOT'] = str(self.root/'dataset/vidstg_test')
        os.environ['VQA_ANNOTATION_PATH'] = str(self.root/'dataset/vidstg_test/annotations/grounded_question_vidstg_test.json')
        sys.path.insert(0, str(self.root/'src/pipeline'))
        import multi_anchor_box_baseline as pipeline
        from google.genai import types
        self.pipeline, self.types = pipeline, types
        self.returned = None
        pipeline.base.call_function = self.hook
        self.cloud = Cloud(config)
        self.state = {'phase': 'starting', 'updated_at': now(), 'target': len(config['targets']),
                      'baseline_completed': config['baseline_completed'], 'project': config['project'],
                      'model': config['model'], 'run_root': str(self.run)}

    def progress(self, **changes):
        self.state.update(changes, updated_at=now())
        save(self.run/'status.json', self.state)
        print(json.dumps(self.state, ensure_ascii=False), flush=True)

    def sample_dir(self, sample):
        return self.run/'samples'/sample

    def source_guard(self):
        for rel, value in self.config['source_sha256'].items():
            if digest((self.root/rel).read_bytes()) != value:
                raise RuntimeError('Planner source changed during batch execution: ' + rel)

    def hook(self, client, model, name, schema, parts):
        if self.returned is None:
            raise CapturedRequest(name, schema, parts)
        args, metadata = self.returned
        if metadata['function_name'] != name:
            raise ValueError('Wrong response bound to planner function')
        return args, metadata

    def result(self, sample, stage):
        return read(self.sample_dir(sample)/(stage+'.json'))

    def wave_key(self, stage):
        attempt = self.config.get('wave_attempts', {}).get(stage, 0)
        return f'{stage}_retry{attempt}' if attempt else stage

    def wave_dir(self, stage):
        return self.run/'waves'/self.wave_key(stage)

    def noun(self, sample):
        original = self.result(sample, 'noun')
        a = self.pipeline.base.NounPhraseResponse.model_validate(original['args']).model_dump(mode='json')
        noun, reasoning = a['answer_noun_phrase'], a['reasoning']
        critic = self.result(sample, 'critic')
        v = critic['args']
        corrected = (v.get('corrected_noun_phrase') or '').strip()
        if v.get('verdict') == 'wrong' and corrected:
            reasoning = f"[critic override of '{noun}'] " + str(v.get('reasoning', ''))
            noun = corrected
        metadata = {**original['metadata'], 'noun_verification': {**v, 'call_metadata': critic['metadata']}}
        return noun, reasoning, a['confidence'], metadata

    def selection(self, sample):
        obj = self.result(sample, 'selection')
        p = self.pipeline
        selection = p.VidSTGAnchorSelectionResponse.model_validate(obj['args'])
        info = read(self.sample_dir(sample)/'prepared.json')
        selection, metadata = p.align_anchor_selection(selection, p.base.stride_frames(info['num_frames'], 10), 3, obj['metadata'])
        selection = p.clamp_grounding_segment(selection, info['num_frames'])
        return selection, metadata

    def invoke(self, target, stage, anchor_frame=None, returned=None):
        p = self.pipeline
        sample = target['sample']
        info = read(self.sample_dir(sample)/'prepared.json')
        self.returned = returned
        common = (None, self.config['model'], Path(info['proxy_path']))
        try:
            if stage == 'noun':
                return p.request_noun_phrase(*common, target['question'], 4., task='vidstg')
            if stage == 'critic':
                candidate = p.base.NounPhraseResponse.model_validate(self.result(sample, 'noun')['args']).answer_noun_phrase
                return p.verify_noun_phrase(*common, target['question'], candidate, 4., task='vidstg')
            if stage == 'selection':
                phrase, _, _, _ = self.noun(sample)
                return p.request_anchor_selection(*common, target['question'], phrase,
                         p.base.stride_frames(info['num_frames'], 10), 3, 4., task='vidstg', num_frames=info['num_frames'])
            if stage == 'boxes':
                selection, _ = self.selection(sample)
                anchor = next(a for a in selection.anchors if a.frame_idx == anchor_frame)
                phrase, _, _, _ = self.noun(sample)
                image = self.sample_dir(sample)/'anchor_frames'/f'frame_{anchor_frame:05d}.png'
                return p.request_anchor_boxes(*common, image, target['question'], phrase,
                         anchor, selection.targets, 4., task='vidstg')
            raise ValueError(stage)
        finally:
            self.returned = None

    def prepare_sample(self, target):
        sample = target['sample']; path = self.sample_dir(sample)/'prepared.json'
        if path.exists():
            return read(path)
        p = self.pipeline
        old = Path(self.config['plan_root'])/sample/'gemini_planning_proxy'
        proxy = old/'planning_video.mp4'; meta_path = old/'frame_map.json'
        valid = False
        if proxy.exists() and meta_path.exists():
            meta = read(meta_path)
            probe = p.base.probe_video(proxy)
            valid = (meta.get('max_frames') == 192 and meta.get('max_dimension') == 512
                     and probe['num_frames'] == meta.get('proxy_frame_count'))
        if not valid:
            video = self.root/'dataset/vidstg_test/videos'/(target['video_id']+'.mp4')
            proxy, meta = p.base.build_planning_proxy_streaming(video, self.sample_dir(sample), 192, 512)
        data = proxy.read_bytes()
        uri = self.cloud.upload(f"{self.config['gcs_prefix']}/media/{sample}/planning_video.mp4", data, 'video/mp4')
        info = {'proxy_path': str(proxy), 'proxy_sha256': digest(data), 'video_uri': uri,
                'num_frames': meta['original_frame_count'], 'source_fps': meta['source_fps'],
                'proxy_metadata': meta, 'image_uri_by_sha256': {}}
        save(path, info)
        return info

    def prepare_box_media(self, target):
        sample = target['sample']; info_path = self.sample_dir(sample)/'prepared.json'
        info = read(info_path); selection, _ = self.selection(sample)
        video = self.root/'dataset/vidstg_test/videos'/(target['video_id']+'.mp4')
        wanted = [a.frame_idx for a in selection.anchors]
        images = self.pipeline.base.load_video_frames_at(video, wanted)
        for frame, image in images.items():
            path = self.sample_dir(sample)/'anchor_frames'/f'frame_{frame:05d}.png'
            self.pipeline.base.save_rgb(image, path)
            data = path.read_bytes()
            uri = self.cloud.upload(f"{self.config['gcs_prefix']}/media/{sample}/frame_{frame:05d}.png", data, 'image/png')
            info['image_uri_by_sha256'][digest(data)] = uri
        save(info_path, info)

    def wire_request(self, sample, capture):
        info = read(self.sample_dir(sample)/'prepared.json')
        parts = []
        for part in capture.parts:
            if part.inline_data is not None:
                data = part.inline_data.data; mime = part.inline_data.mime_type
                obj = part.model_copy(update={'inline_data': None}).model_dump(mode='json', by_alias=True, exclude_none=True)
                if mime == 'video/mp4':
                    if digest(data) != info['proxy_sha256']:
                        raise ValueError('Planning proxy changed after GCS upload')
                    uri = info['video_uri']
                else:
                    uri = info['image_uri_by_sha256'][digest(data)]
                obj['fileData'] = {'fileUri': uri, 'mimeType': mime}
            else:
                obj = part.model_dump(mode='json', by_alias=True, exclude_none=True)
            parts.append(obj)
        cfg = self.pipeline.base.function_config(capture.name, capture.schema).model_dump(mode='json', by_alias=True, exclude_none=True)
        for tool in cfg.get('tools', []):
            for declaration in tool.get('functionDeclarations', []):
                if 'parametersJsonSchema' in declaration:
                    declaration['parametersJsonSchema'] = inline_json_schema(declaration['parametersJsonSchema'])
        request = {'contents': [{'role': 'user', 'parts': parts}],
                   'tools': cfg.pop('tools'), 'toolConfig': cfg.pop('toolConfig')}
        if cfg:
            request['generationConfig'] = cfg
        return request

    def prepare_wave(self, stage):
        directory = self.wave_dir(stage)
        manifest_path = directory/'manifest.json'
        if manifest_path.exists():
            return read(manifest_path)
        previous = STAGES[STAGES.index(stage)-1] if stage != 'noun' else None
        targets = [t for t in self.config['targets'] if previous is None or (self.sample_dir(t['sample'])/(previous+'.json')).exists()]
        if stage == 'boxes':
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(self.prepare_box_media, targets))
        manifest = []
        lines = []
        for target in targets:
            sample = target['sample']
            frames = [a.frame_idx for a in self.selection(sample)[0].anchors] if stage == 'boxes' else [None]
            for frame in frames:
                try:
                    self.invoke(target, stage, frame)
                except CapturedRequest as capture:
                    request = self.wire_request(sample, capture)
                    signature = request_signature(request)
                    manifest.append({'sample': sample, 'stage': stage, 'anchor_frame': frame,
                                     'function_name': capture.name, 'signature': signature})
                    lines.append(json.dumps({'request': request}, ensure_ascii=False))
                else:
                    raise RuntimeError('Request construction unexpectedly returned a result')
        if len({m['signature'] for m in manifest}) != len(manifest):
            raise ValueError('Batch requests lack unique source-media bindings')
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'requests.jsonl').write_text('\n'.join(lines)+'\n' if lines else '')
        save(manifest_path, manifest)
        return manifest

    def parent_url(self):
        return f"https://aiplatform.googleapis.com/v1/projects/{self.config['project']}/locations/global/batchPredictionJobs"

    def submit(self, stage, manifest):
        directory = self.wave_dir(stage)
        receipt = directory/'submission.json'
        display = self.config['run_id']+'-'+self.wave_key(stage)
        if receipt.exists():
            recorded = read(receipt)
            if recorded.get('name'):
                return recorded
            # Reconcile an uncertain POST before any possibility of resubmission.
            jobs = self.cloud.request('GET', self.parent_url(), params={'filter': f'displayName="{display}"'}).json().get('batchPredictionJobs', [])
            if len(jobs) == 1:
                save(receipt, jobs[0]); return jobs[0]
            raise RuntimeError('Submission outcome uncertain; no duplicate job was created. ' + str(receipt))
        data = (directory/'requests.jsonl').read_bytes()
        if not manifest:
            raise RuntimeError('No eligible requests in wave ' + stage)
        if len(data) >= 1_000_000_000:
            raise ValueError('Batch JSONL exceeds 1 GB')
        source = self.cloud.upload(f"{self.config['gcs_prefix']}/waves/{self.wave_key(stage)}/requests.jsonl", data, 'application/jsonl')
        destination = f"gs://{self.config['bucket']}/{self.config['gcs_prefix']}/waves/{self.wave_key(stage)}/output/"
        body = {'displayName': display, 'model': 'publishers/google/models/'+self.config['model'],
                'inputConfig': {'instancesFormat': 'jsonl', 'gcsSource': {'uris': [source]}},
                'outputConfig': {'predictionsFormat': 'jsonl', 'gcsDestination': {'outputUriPrefix': destination}},
                'labels': {'pipeline': 'vidstg-planning', 'stage': stage}}
        save(directory/'submission_request.json', body)
        save(receipt, {'status': 'creating', 'displayName': display, 'created_at': now(), 'request_count': len(manifest)})
        job = self.cloud.request('POST', self.parent_url(), json=body).json()
        save(receipt, job)
        self.progress(phase=stage+'_submitted', current_stage=stage, job_name=job['name'], request_count=len(manifest), job_state=job.get('state'))
        return job

    def await_job(self, stage, job):
        while True:
            job = self.cloud.request('GET', 'https://aiplatform.googleapis.com/v1/'+job['name']).json()
            save(self.wave_dir(stage)/'job_latest.json', job)
            self.progress(phase=stage+'_waiting', current_stage=stage, job_name=job['name'],
                          job_state=job.get('state'), completion_stats=job.get('completionStats', {}))
            if job.get('state') in TERMINAL:
                return job
            time.sleep(45)

    def collect(self, stage, job, manifest):
        directory = self.wave_dir(stage)
        complete_path = directory/'collected.json'
        if complete_path.exists():
            return read(complete_path)
        expected = {m['signature']: m for m in manifest}
        targets = {t['sample']: t for t in self.config['targets']}
        outputs = self.cloud.list_output(f"{self.config['gcs_prefix']}/waves/{self.wave_key(stage)}/output/")
        failures, accepted, seen = [], [], set()
        for item in outputs:
            if not item['name'].endswith('.jsonl'):
                continue
            raw = self.cloud.download(item['name'])
            (directory/Path(item['name']).name).write_bytes(raw)
            for line in raw.decode().splitlines():
                if not line.strip(): continue
                row = json.loads(line)
                signature = request_signature(row.get('request') or {})
                if signature not in expected:
                    raise ValueError('Batch output cannot be matched to an exact submitted input')
                if signature in seen:
                    raise ValueError('Duplicate batch output for one submitted request')
                seen.add(signature); record = expected[signature]
                sample = record['sample']; frame = record['anchor_frame']
                try:
                    args, response = extract_call(row, record['function_name'])
                    usage_obj = self.types.GenerateContentResponse.model_validate(response).usage_metadata
                    usage = usage_obj.model_dump(mode='json') if usage_obj else None
                    metadata = {'model': self.config['model'], 'function_name': record['function_name'],
                                'usage_metadata': usage, 'model_version': response.get('modelVersion'),
                                'batch': {'job_name': job['name'], 'signature': signature, 'processed_time': row.get('processed_time')}}
                    if stage == 'critic':
                        if (args.get('verdict') not in ['correct','wrong'] or not isinstance(args.get('reasoning'), str)
                                or not isinstance(args.get('expected_instance_count'), int) or args['expected_instance_count'] < 1):
                            raise ValueError('Invalid critic verification response')
                        if args.get('corrected_noun_phrase') is not None and not isinstance(args['corrected_noun_phrase'], str):
                            raise ValueError('Invalid critic correction type')
                        if args.get('evidence_frames') is not None and (not isinstance(args['evidence_frames'], list)
                                or any(not isinstance(f, int) for f in args['evidence_frames'])):
                            raise ValueError('Invalid critic evidence-frame type')
                    self.invoke(targets[sample], stage, frame, (args, metadata))
                    suffix = f'boxes_{frame:05d}' if stage == 'boxes' else stage
                    save(self.sample_dir(sample)/(suffix+'.json'), {'args': args, 'metadata': metadata})
                    accepted.append(record)
                except (ValueError, TypeError, KeyError) as exc:
                    failures.append({**record, 'error': str(exc), 'response': row})
        for signature in set(expected)-seen:
            failures.append({**expected[signature], 'error': 'No output row returned', 'job_error': job.get('error')})
        if stage == 'boxes':
            for target in self.config['targets']:
                sample=target['sample']
                if not (self.sample_dir(sample)/'selection.json').exists(): continue
                selection,_=self.selection(sample)
                if all((self.sample_dir(sample)/f'boxes_{a.frame_idx:05d}.json').exists() for a in selection.anchors):
                    save(self.sample_dir(sample)/'boxes.json', {'status': 'complete'})
        result = {'stage': stage, 'job_name': job['name'], 'requests': len(manifest), 'accepted': len(accepted), 'failed': len(failures), 'failures': failures}
        save(complete_path, result)
        if not accepted:
            raise RuntimeError(f'No valid {stage} responses; see {complete_path}')
        return result

    def finalize(self, target):
        sample=target['sample'];p=self.pipeline
        if not (self.sample_dir(sample)/'boxes.json').exists(): return False
        destination=Path(self.config['plan_root'])/sample
        if (destination/'multi_anchor_plan.json').exists(): return False
        info=read(self.sample_dir(sample)/'prepared.json');selection,selection_meta=self.selection(sample)
        indices,sampling=p.segment_boundary_dense_frames(info['num_frames'],10,selection,15,
                         base_sampling_metadata={'source_fps':info['source_fps'],'sam_target_fps':None})
        anchors=[]
        for anchor in selection.anchors:
            result=self.result(sample,f'boxes_{anchor.frame_idx:05d}')
            boxes=p.AnchorBoxesResponse.model_validate(result['args'])
            anchors.append(p.SavedAnchor(frame_idx=anchor.frame_idx,sam_frame_idx=indices.index(anchor.frame_idx),
                           obj_ids=anchor.obj_ids,reason=anchor.reason,boxes=boxes.boxes,
                           exact_frame_path=str(self.sample_dir(sample)/'anchor_frames'/f'frame_{anchor.frame_idx:05d}.png'),
                           call_metadata=result['metadata']))
        phrase,reason,confidence,noun_meta=self.noun(sample)
        plan=p.SavedMultiAnchorPlan(video_id=target['video_id'],question_id=target['question_id'],question=target['question'],
            answer_noun_phrase=phrase,noun_reasoning=reason,noun_confidence=confidence,
            decision_summary=selection.decision_summary,first_visible_frame_idx=selection.first_visible_frame_idx,
            last_visible_frame_idx=selection.last_visible_frame_idx,visibility_boundary_reason=selection.visibility_boundary_reason,
            targets=selection.targets,anchors=anchors,sam_stride=10,sam_target_fps=None,required_frame_stride=None,
            boundary_radius=15,max_anchor_frames=3,anchor_policy='gemini',model=self.config['model'],gemini_video_fps=4.,
            task='vidstg',grounding_segment=selection.grounding_segment,
            call_metadata={'noun_phrase':noun_meta,'anchor_selection':selection_meta,
                           'gemini_planning_proxy':info['proxy_metadata'],
                           'batch_run':{'run_root':str(self.run),'run_id':self.config['run_id']}})
        save(destination/'anchor_selection.json',selection.model_dump(mode='json'))
        save(destination/'rl_episode.json',p.make_rl_episode(p.base.stride_frames(info['num_frames'],10),selection,'gemini',3))
        save(destination/'sam_frame_map.json',{**sampling,'num_original_frames':info['num_frames'],
             'source_fps':info['source_fps'],'sam_target_fps':None,'required_frame_stride':0,
             'anchors':[a.model_dump(mode='json') for a in anchors],
             'sam_to_original_frame':{str(i):f for i,f in enumerate(indices)}})
        # Completion marker is written last. Existing completed plans are never replaced.
        save(destination/'multi_anchor_plan.json',plan.model_dump(mode='json'))
        return True

    def run_all(self):
        self.source_guard();self.cloud.ensure_bucket()
        self.progress(phase='preparing_media', prepared=0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(self.prepare_sample,t) for t in self.config['targets']]
            for count,future in enumerate(concurrent.futures.as_completed(futures),1):
                future.result()
                if count%50==0 or count==len(futures): self.progress(prepared=count)
        summaries={}
        for stage in STAGES:
            self.source_guard();self.progress(phase=stage+'_preparing',current_stage=stage)
            manifest=self.prepare_wave(stage)
            job=self.submit(stage,manifest)
            job=self.await_job(stage,job)
            summaries[stage]=self.collect(stage,job,manifest)
            self.progress(phase=stage+'_collected',wave_accepted=summaries[stage]['accepted'],wave_failed=summaries[stage]['failed'])
        completed=sum(self.finalize(t) for t in self.config['targets'])
        plan_root=Path(self.config['plan_root'])
        lines=[]
        for line in (self.root/'dataset/vidstg_test/subset_declarative.tsv').read_text().splitlines():
            if not line.strip():continue
            v,q=line.split('\t')
            if (plan_root/f'{v}_q{q}'/'multi_anchor_plan.json').exists():lines.append(line)
        planned=self.root/'outputs/vidstg_test/planned_subset_declarative.tsv'
        planned.write_text('\n'.join(lines)+'\n')
        save(self.run/'summary.json',{'finished_at':now(),'newly_written_plans':completed,
             'total_declarative_plans':len(lines),'target':len(self.config['targets']),'waves':summaries})
        self.progress(phase='complete' if len(lines)==4610 else 'complete_with_failures',
                      new_plans=completed,total_declarative_plans=len(lines),remaining_declarative=4610-len(lines))


def initialize(args):
    root=Path(args.release_root).resolve();plan_root=root/'outputs/vidstg_test/plans'
    annotations=read(root/'dataset/vidstg_test/annotations/grounded_question_vidstg_test.json')
    targets=[];baseline=0
    for line in (root/'dataset/vidstg_test/subset_declarative.tsv').read_text().splitlines():
        if not line.strip():continue
        video,qid=line.split('\t');sample=f'{video}_q{qid}'
        if (plan_root/sample/'multi_anchor_plan.json').exists():baseline+=1;continue
        item=next(q for q in annotations[video]['grounded_question'] if str(q['id'])==qid)
        assert item['sentence_type']=='declarative'
        targets.append({'sample':sample,'video_id':video,'question_id':qid,'question':item['question']})
    run=Path(args.run_root).resolve()
    if (run/'config.json').exists():raise RuntimeError('Run already initialized')
    sources=['src/pipeline/multi_anchor_box_baseline.py','src/pipeline/box_baseline.py']
    config={'run_id':run.name.replace('_','-'),'run_root':str(run),'release_root':str(root),
            'plan_root':str(plan_root),'project':args.project,'bucket':args.bucket,
            'gcs_prefix':'runs/'+run.name,'model':'gemini-3.8-flash','created_at':now(),
            'baseline_completed':baseline,'targets':targets,
            'source_sha256':{n:digest((root/n).read_bytes()) for n in sources}}
    save(run/'config.json',config)
    for n in sources:
        dst=run/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/n,dst)
    (run/'subset_remaining.tsv').write_text(''.join(f"{t['video_id']}\t{t['question_id']}\n" for t in targets))
    print(json.dumps({'config':str(run/'config.json'),'baseline_completed':baseline,'target':len(targets),'project':args.project,'bucket':args.bucket}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init');init.add_argument('--release-root',default='.')
    init.add_argument('--project',required=True);init.add_argument('--bucket',required=True);init.add_argument('--run-root',required=True)
    run=sub.add_parser('run');run.add_argument('--config',required=True)
    args=parser.parse_args()
    if args.command=='init':initialize(args);return
    config=read(args.config);lock=Path(config['plan_root'])/'.batch_planning.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        planner=Planner(config)
        try:planner.run_all()
        except Exception as exc:
            planner.progress(phase='stopped_error',error_type=type(exc).__name__,error=str(exc))
            raise


if __name__=='__main__':main()
