#!/usr/bin/env python3
"""Durable Vertex Batch transport for the existing visible VidSTG Sparse stage.

Sampling is 1 FPS on each clip's native frame clock, with no duration/frame cap.
Input images are decoder-exact original frames. Original plans and online Sparse
code remain unchanged. No GT is loaded. Calls retain temporal image context.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import quote

from batch_plan import Cloud, TERMINAL, digest, inline_json_schema, now, read, request_signature, save


def sample_frames(num_frames, fps, sample_fps=1.):
    if num_frames < 1 or not math.isfinite(fps) or fps <= 0 or not 0 < sample_fps <= fps:
        raise ValueError('Invalid native frame count or FPS')
    output = []
    k = 0
    while True:
        frame = math.floor(k * fps / sample_fps + 0.5)
        if frame >= num_frames:
            break
        output.append(frame)
        k += 1
    assert output == sorted(set(output))
    return output


def batches(frames, size):
    return [frames[i:i+size] for i in range(0, len(frames), size)]


def response_json(row):
    if row.get('status'):
        raise ValueError('Batch row status: ' + str(row['status']))
    candidates = (row.get('response') or {}).get('candidates') or []
    if not candidates:
        raise ValueError('No response candidate')
    content = candidates[0].get('content') or {}
    text = ''.join(p.get('text', '') for p in content.get('parts', []) if not p.get('thought'))
    return json.loads(text)


class SparseBatch:
    def __init__(self, config):
        self.c = config
        self.run = Path(config['run_root'])
        self.root = Path(config['release_root'])
        sys.path.insert(0, str(self.root/'src/sparse_fusion'))
        import generate_sparse as sparse
        self.s = sparse
        self.cloud = Cloud(config)
        self.lock = threading.Lock()
        self.state = {'phase': 'starting', 'target_samples': len(config['targets']),
                      'sampling_fps': config['sampling_fps'], 'frames_per_call': config['frames_per_call'],
                      'max_frames_per_video': None, 'box_mode': 'visible',
                      'output_root': config['output_root'], 'updated_at': now()}

    def progress(self, **kwargs):
        self.state.update(kwargs, updated_at=now())
        save(self.run/'status.json', self.state)
        print(json.dumps(self.state), flush=True)

    def guard(self):
        for rel, sha in self.c['source_sha256'].items():
            if digest((self.root/rel).read_bytes()) != sha:
                raise RuntimeError('Sparse source changed: ' + rel)

    def upload_frame(self, clip, frame):
        p = self.run/'media'/clip/f'frame_{frame:06d}.jpg'
        data = p.read_bytes()
        return str(frame), self.cloud.upload(
            f"{self.c['gcs_prefix']}/media/{clip}/frame_{frame:06d}.jpg", data, 'image/jpeg')

    def prepare_clip(self, clip, info, upload_pool):
        directory = self.run/'media'/clip
        marker = directory/'manifest.json'
        source = self.root/'dataset/vidstg_test/videos'/f'{clip}.mp4'
        stat = source.stat()
        identity = {'path': str(source), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        frames = sample_frames(info['num_frames'], info['fps'], self.c['sampling_fps'])
        if marker.exists():
            cached = read(marker)
            if cached['source'] != identity or cached['frame_indices'] != frames:
                raise ValueError('Cached source or sampling changed: ' + clip)
            if cached.get('uploaded'):
                return cached
        directory.mkdir(parents=True, exist_ok=True)
        extraction = directory/'extraction.json'
        if not extraction.exists():
            expression = '+'.join(f'eq(n\\,{f})' for f in frames)
            command = ['ffmpeg','-y','-nostdin','-hide_banner','-loglevel','error','-threads','1',
                       '-i',str(source),'-vf','select='+expression+',settb=1/1,setpts=N','-r','1',
                       '-q:v','2','-start_number','0',str(directory/'selected_%06d.jpg')]
            subprocess.run(command, check=True, capture_output=True)
            selected = sorted(directory.glob('selected_*.jpg'))
            if len(selected) != len(frames):
                raise ValueError(f'Decoded frame count mismatch {clip}: {len(selected)} != {len(frames)}')
            for path, frame in zip(selected, frames):
                path.replace(directory/f'frame_{frame:06d}.jpg')
            save(extraction, {'source': identity, 'frame_indices': frames})
        cached = {'source': identity, 'frame_indices': frames, 'video': info,
                  'sampling_fps': self.c['sampling_fps'], 'max_frames_per_video': None}
        pending = [upload_pool.submit(self.upload_frame, clip, frame) for frame in frames]
        cached['image_uris'] = dict(p.result() for p in futures.as_completed(pending))
        cached['uploaded'] = True
        save(marker, cached)
        return cached

    def prepare_media(self):
        self.cloud.ensure_bucket()
        clips = self.c['clips']
        failed = []
        self.progress(phase='preparing_media', prepared_clips=0, total_clips=len(clips))
        with futures.ThreadPoolExecutor(max_workers=self.c['upload_workers']) as upload_pool:
            with futures.ThreadPoolExecutor(max_workers=self.c['decode_workers']) as decode_pool:
                pending = {decode_pool.submit(self.prepare_clip, v, i, upload_pool): v for v, i in clips.items()}
                for count, task in enumerate(futures.as_completed(pending), 1):
                    try:
                        task.result()
                    except Exception as exc:
                        failed.append({'clip': pending[task], 'error': str(exc)})
                        save(self.run/'media_failures.json', failed)
                    if count % 25 == 0 or count == len(pending):
                        self.progress(prepared_clips=count-len(failed), failed_clips=len(failed))
        if failed:
            raise RuntimeError(f'{len(failed)} clips failed preparation; resume only missing media')

    def missing_plans(self):
        return [t['sample'] for t in self.c['targets']
                if not (Path(self.c['plan_root'])/t['sample']/'multi_anchor_plan.json').exists()]

    def prepare_requests(self):
        directory = self.run/'batch'
        marker = directory/'manifest.json'
        if marker.exists():
            return read(marker)
        while self.missing_plans():
            self.progress(phase='waiting_for_anchor_repair', missing_plans=self.missing_plans())
            time.sleep(30)
        self.guard()
        directory.mkdir(parents=True, exist_ok=True)
        manifest = []
        schema = inline_json_schema(self.s.SparseResponse.model_json_schema())
        with (directory/'requests.jsonl').open('w') as out:
            for t in self.c['targets']:
                sample = t['sample']
                plan_path = Path(self.c['plan_root'])/sample/'multi_anchor_plan.json'
                plan = read(plan_path)
                # Freeze inputs before inference; never read mutable plans when collecting.
                snapshot = self.run/'plans'/sample/'multi_anchor_plan.json'
                save(snapshot, plan)
                media = read(self.run/'media'/t['video_id']/'manifest.json')
                info = media['video']
                for index, group in enumerate(batches(media['frame_indices'], self.c['frames_per_call'])):
                    call_schema = json.loads(json.dumps(schema))
                    frame_array = call_schema['properties']['frames']
                    frame_array.update(minItems=len(group), maxItems=len(group))
                    frame_array['items']['properties']['frame_idx'].update(minimum=min(group), maximum=max(group))
                    objects = frame_array['items']['properties']['objects']
                    known_ids = sorted(int(t['obj_id']) for t in plan['targets'])
                    objects['maxItems'] = min(8, len(known_ids))
                    objects['items']['properties']['object_id'].update(minimum=min(known_ids), maximum=max(known_ids))
                    prompt = self.s.build_visible_prompt(plan, group, info, 0)
                    prompt = prompt.replace('These are fixed 0-frame-grid positions.',
                        'These are exact original frame IDs sampled at 1 FPS on the native video clock.')
                    parts = [{'text': prompt}] + [
                        {'fileData': {'fileUri': media['image_uris'][str(f)], 'mimeType': 'image/jpeg'}} for f in group]
                    request = {'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': {
                        'temperature': 0., 'thinkingConfig': {'thinkingLevel': 'HIGH'},
                        'maxOutputTokens': 32768, 'responseMimeType': 'application/json',
                        'responseJsonSchema': call_schema}}
                    signature = request_signature(request)
                    manifest.append({'sample': sample, 'video_id': t['video_id'], 'question_id': t['question_id'],
                                     'call_index': index, 'frame_indices': group, 'signature': signature})
                    out.write(json.dumps({'request': request}, ensure_ascii=False)+'\n')
        if len(set(x['signature'] for x in manifest)) != len(manifest):
            raise ValueError('Duplicate request signatures')
        size = (directory/'requests.jsonl').stat().st_size
        if size >= 1_000_000_000 or len(manifest) > 200000:
            raise ValueError(f'Input exceeds Batch API limits: {size} bytes / {len(manifest)} requests')
        save(marker, manifest)
        self.progress(phase='requests_prepared', request_count=len(manifest), jsonl_bytes=size,
                      sampled_frames=sum(len(x['frame_indices']) for x in manifest))
        return manifest

    def parent_url(self):
        return f"https://aiplatform.googleapis.com/v1/projects/{self.c['project']}/locations/global/batchPredictionJobs"

    def submit(self, manifest):
        directory = self.run/'batch'
        receipt = directory/'submission.json'
        display = self.run.name.replace('_', '-')
        if receipt.exists():
            job = read(receipt)
            if job.get('name'):
                return job
            jobs = self.cloud.request('GET', self.parent_url(), params={'filter': f'displayName="{display}"'}).json().get('batchPredictionJobs', [])
            if len(jobs) != 1:
                raise RuntimeError('Uncertain prior submission; refusing to create a duplicate')
            save(receipt, jobs[0])
            return jobs[0]
        source = self.cloud.upload(f"{self.c['gcs_prefix']}/batch/requests.jsonl",
                                   (directory/'requests.jsonl').read_bytes(), 'application/jsonl')
        body = {'displayName': display, 'model': 'publishers/google/models/'+self.c['model'],
                'inputConfig': {'instancesFormat': 'jsonl', 'gcsSource': {'uris': [source]}},
                'outputConfig': {'predictionsFormat': 'jsonl', 'gcsDestination': {
                    'outputUriPrefix': f"gs://{self.c['bucket']}/{self.c['gcs_prefix']}/batch/output/"}},
                'labels': {'pipeline': 'vidstg-sparse', 'mode': 'visible', 'fps': '1'}}
        save(directory/'submission_request.json', body)
        save(receipt, {'state': 'creating', 'created_at': now(), 'displayName': display})
        job = self.cloud.request('POST', self.parent_url(), json=body).json()
        save(receipt, job)
        self.progress(phase='submitted', job_name=job['name'], job_state=job.get('state'), request_count=len(manifest))
        return job

    def collect(self, job, manifest):
        directory = self.run/'batch'
        accepted = directory/'accepted'
        accepted.mkdir(exist_ok=True)
        by_sig = {m['signature']: m for m in manifest}
        seen, failures = set(), []
        output_dir = (job.get('outputInfo') or {}).get('gcsOutputDirectory')
        prefix = self.c['gcs_prefix']+'/batch/output/'
        if output_dir:
            prefix = output_dir.removeprefix(f"gs://{self.c['bucket']}/").rstrip('/')+'/'
        for obj in self.cloud.list_output(prefix):
            if not obj['name'].endswith('.jsonl'):
                continue
            raw = self.cloud.download(obj['name'])
            (directory/Path(obj['name']).name).write_bytes(raw)
            for line in raw.decode().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                signature = request_signature(row.get('request') or {})
                if signature not in by_sig or signature in seen:
                    raise ValueError('Unbound or duplicate output row')
                seen.add(signature)
                record = by_sig[signature]
                try:
                    parsed = self.s.SparseResponse.model_validate(response_json(row))
                    plan = read(self.run/'plans'/record['sample']/'multi_anchor_plan.json')
                    self.s.validate_response(parsed, plan, record['frame_indices'])
                    save(accepted/(signature+'.json'), {'response': parsed.model_dump(mode='json'),
                         'usage_metadata': (row.get('response') or {}).get('usageMetadata'), 'batch': {
                             'job_name': job['name'], 'signature': signature}})
                except (ValueError, KeyError, TypeError) as exc:
                    failures.append({**record, 'error': str(exc), 'raw_output': row})
        failures.extend({**by_sig[sig], 'error': 'No output row returned'} for sig in set(by_sig)-seen)
        save(directory/'collection.json', {'requests': len(manifest), 'failed': len(failures),
             'accepted': len(manifest)-len(failures), 'failures': failures, 'job_name': job['name']})
        complete = 0
        by_sample = {}
        for m in manifest:
            by_sample.setdefault(m['sample'], []).append(m)
        for sample, calls in by_sample.items():
            if any(not (accepted/(m['signature']+'.json')).exists() for m in calls):
                continue
            plan = read(self.run/'plans'/sample/'multi_anchor_plan.json')
            t = calls[0]
            media = read(self.run/'media'/t['video_id']/'manifest.json')
            output = Path(self.c['output_root'])/sample
            output.mkdir(parents=True, exist_ok=True)
            preds, visibility = {}, {'visible': 0, 'partially_occluded': 0, 'fully_occluded': 0}
            for m in calls:
                saved = read(accepted/(m['signature']+'.json'))
                save(output/'calls'/f"call_{m['call_index']:03d}.json", {'attempts': [{
                    'status': 'ok', 'frame_indices': m['frame_indices'], 'model': self.c['model'], **saved}]})
                for frame in saved['response']['frames']:
                    objects = []
                    for obj in frame['objects']:
                        if obj['visibility'] == 'fully_occluded':
                            continue
                        visibility[obj['visibility']] += 1
                        objects.append({'object_id': obj['object_id'], 'visibility': obj['visibility'],
                            'confidence': obj['confidence'], 'box_2d_yxyx_1000': obj['box_2d'],
                            'xyxy_normalized': self.s.yxyx_1000_to_xyxy_normalized(obj['box_2d'])})
                    preds[str(frame['frame_idx'])] = objects
            # frame_stride is nominal for legacy consumers; frame_indices are authoritative.
            nominal = max(1, round(media['video']['fps']/self.c['sampling_fps']))
            metadata = {'sampling_fps': self.c['sampling_fps'], 'frame_stride': nominal,
                        'frame_stride_is_nominal': True, 'frame_indices': media['frame_indices'],
                        'sampling_method': 'nearest_original_frame_to_each_integer_second',
                        'box_mode': 'visible', 'grounding_margin': -1, 'max_frames_per_video': None}
            save(output/'manifest.json', {'sample': sample, 'video_id': t['video_id'],
                'question_id': t['question_id'], 'model': self.c['model'], **metadata,
                'frames_per_call': self.c['frames_per_call'], 'thinking_level': 'high',
                'prompt_contract': 'visible_extent_v1+native_fps_sampling',
                'plan_path': str(self.run/'plans'/sample/'multi_anchor_plan.json'), 'video': media['video']})
            save(output/'sparse_predictions.json', {'video_id': t['video_id'], 'question_id': t['question_id'],
                'answer_noun_phrase': plan.get('answer_noun_phrase', ''), **metadata,
                'targets': self.s.compact_plan(plan)['targets'], 'predictions_by_frame': preds})
            save(output/'run_result.json', {'status': 'complete', 'sample': sample, 'model': self.c['model'],
                'num_sparse_frames': len(preds), 'num_calls': len(calls),
                'num_boxes': sum(map(len, preds.values())), 'visibility_counts': visibility})
            complete += 1
        save(self.run/'summary.json', {'completed_at': now(), 'complete_samples': complete,
             'target_samples': len(self.c['targets']), 'failed_requests': len(failures),
             'output_root': self.c['output_root'], 'job_name': job['name']})
        self.progress(phase='complete' if complete == len(self.c['targets']) else 'complete_with_failures',
                      complete_samples=complete, failed_requests=len(failures))

    def run_all(self):
        self.guard()
        self.prepare_media()
        manifest = self.prepare_requests()
        self.guard()
        job = self.submit(manifest)
        while True:
            job = self.cloud.request('GET', 'https://aiplatform.googleapis.com/v1/'+job['name']).json()
            save(self.run/'batch/job_latest.json', job)
            self.progress(phase='waiting', job_name=job['name'], job_state=job.get('state'),
                          completion_stats=job.get('completionStats', {}))
            if job.get('state') in TERMINAL:
                break
            time.sleep(45)
        self.collect(job, manifest)


def initialize(args):
    root, run = Path(args.release_root).resolve(), Path(args.run_root).resolve()
    if (run/'config.json').exists():
        raise RuntimeError('Run already initialized')
    original = read(root/'outputs/vidstg_test/active_batch_planning.json')
    targets = []
    for line in (root/'dataset/vidstg_test/subset_declarative.tsv').read_text().splitlines():
        if line.strip():
            v, q = line.split('\t')
            targets.append({'video_id': v, 'question_id': q, 'sample': f'{v}_q{q}'})
    assert len(targets) == 4610 and len({x['sample'] for x in targets}) == 4610
    clips = read(root/'dataset/vidstg_test/vidstg_clips.json')
    used = {v: clips[v] for v in sorted({x['video_id'] for x in targets})}
    source = ['src/sparse_fusion/generate_sparse.py', 'src/sparse_fusion/sparse_common.py']
    c = {'release_root': str(root), 'run_root': str(run), 'output_root': str(run/'sparse'),
         'plan_root': str(root/'outputs/vidstg_test/plans'), 'project': original['project'],
         'bucket': original['bucket'], 'model': original['model'], 'gcs_prefix': 'runs/'+run.name,
         'sampling_fps': 1., 'max_frames_per_video': None, 'frames_per_call': args.frames_per_call,
         'decode_workers': args.decode_workers, 'upload_workers': args.upload_workers,
         'targets': targets, 'clips': used, 'created_at': now(),
         'source_sha256': {p: digest((root/p).read_bytes()) for p in source}}
    save(run/'config.json', c)
    for p in source:
        dst = run/'source_snapshot'/p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/p, dst)
    (run/'subset.tsv').write_text(''.join(f"{t['video_id']}\t{t['question_id']}\n" for t in targets))
    counts = [len(sample_frames(clips[t['video_id']]['num_frames'], clips[t['video_id']]['fps'])) for t in targets]
    receipt = {'config': str(run/'config.json'), 'run_root': str(run), 'output_root': c['output_root'],
               'target_samples': len(targets), 'unique_clips': len(used), 'sample_frame_pairs': sum(counts),
               'estimated_requests': sum(math.ceil(n/args.frames_per_call) for n in counts),
               'max_frames_in_input_sample': max(counts), 'sampling_fps': 1., 'max_frames_per_video': None}
    save(run/'initialization.json', receipt)
    save(root/'outputs/vidstg_test/active_sparse_batch.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'prepare', 'run'])
    parser.add_argument('--config')
    parser.add_argument('--release-root', default='.')
    parser.add_argument('--run-root')
    parser.add_argument('--frames-per-call', type=int, default=8)
    parser.add_argument('--decode-workers', type=int, default=8)
    parser.add_argument('--upload-workers', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.frames_per_call <= 15:
        parser.error('frames-per-call must be 1..15')
    if args.command == 'init':
        initialize(args)
        return
    config = read(args.config)
    with (Path(config['run_root'])/'.controller.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = SparseBatch(config)
        try:
            if args.command == 'prepare':
                controller.guard()
                controller.prepare_media()
                controller.progress(phase='media_prepared')
            else:
                controller.run_all()
        except Exception as exc:
            controller.progress(phase='stopped_error', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
