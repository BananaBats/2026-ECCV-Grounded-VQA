"""Prepare or resume the native-frame Gemini Direct amodal baseline.

This retains the actual all-frame runner, prompt/schema, adaptive96-frame chunks,
stride30 whole-video context, and 3-frame identity handoff. Calls are dependent
Standard API requests, not an independent one-shot Batch approximation.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from .common import queries,read,save,digest,load_file,LEGACY

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queries',type=Path,required=True);p.add_argument('--video-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--project',default=os.environ.get('GOOGLE_CLOUD_PROJECT'));p.add_argument('--location',default='global');p.add_argument('--sample',action='append');p.add_argument('--max-new-calls',type=int);p.add_argument('--prepare-only',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args(argv);rows=queries(a.queries);rows=[r for r in rows if not a.sample or r['sample'] in a.sample]
 if not rows:raise ValueError('No selected questions')
 if a.dry_run:print(json.dumps({'phase':'dry_run','queries':len(rows),'method':'native-frame-amodal-context+96framechunks+3handoff','API_calls':0}));return 0
 from .common import load_pipeline
 pipeline=load_pipeline('perception');pipeline.base.VIDEO_ROOT=a.video_root.resolve();registered=[]
 for r in rows:
  video=Path(r['video_path']).resolve() if r.get('video_path') else pipeline.base.video_path_for(r['video_id']);meta=pipeline.base.probe_video(video)
  registered.append({**r,'video_path':str(video),'num_frames':int(meta['num_frames']),'width':int(meta['width']),'height':int(meta['height']),'fps':float(meta['fps']),'source_fps':float(meta['fps'])})
 directory=LEGACY/'direct_amodal';runner=load_file(directory/'run.py','_retained_amodal_runner');manifest={'method_revision':runner.core.METHOD_REVISION,'model':'gemini-3.7-flash','project':a.project or 'SET_YOUR_PROJECT','location':a.location,'generation':{'temperature':0,'thinking_level':'high','max_output_tokens':65536},'sampling':{'frame_stride':1,'context_stride':30,'max_chunk_frames':96,'max_inline_bytes':64*1024*1024,'handoff_frames':3},'queries':registered,'source_sha256':{str(directory/'run.py'):digest(directory/'run.py'),str(directory/'core.py'):digest(directory/'core.py')},'inference_reads_GT':False}
 mp=a.output_root/'inference_manifest.json'
 if mp.exists():
  if read(mp)!=manifest:raise ValueError('Existing Direct manifest differs; use another output directory')
 else:save(mp,manifest)
 if a.prepare_only:print(json.dumps({'phase':'prepared','manifest':str(mp),'API_calls':0}));return 0
 if not a.project:raise ValueError('Set --project or GOOGLE_CLOUD_PROJECT to your own Vertex project')
 runner.verify_manifest(a.output_root,manifest,inference=True);r=runner.Runner(a.output_root,manifest,a.max_new_calls)
 try:
  for query in manifest['queries']:r.run_sample(query)
 except runner.CallLimitReached:
  r.status('call_limit_reached');return 0
 merged={}
 for query in registered:
  path=a.output_root/'A30'/query['sample']/'predictions.json'
  if path.exists():
   for v,item in read(path).items():merged.setdefault(v,{'grounded_question':{}})['grounded_question'].update(item['grounded_question'])
 save(a.output_root/'predictions.json',merged);r.status('complete' if not r.failed else 'complete_with_failures');return 1 if r.failed else 0
if __name__=='__main__':raise SystemExit(main())
