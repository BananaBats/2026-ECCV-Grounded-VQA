"""Run saved-plan SAM3 dense tracking with retained experimental code.

No model selection or Gemini request occurs here. One process owns one GPU;
--persistent keeps the predictor in memory while every query opens a new session.
The legacy compact export is preserved, including its historical hold behavior.
"""
from __future__ import annotations
import argparse,json,math,os
from pathlib import Path
from .common import PROFILES,load_pipeline,configure_media,read,save,simple_rows,object_sha
from .sampling_vidstg import sample_frames as vidstg_grid
from .svag_dense_sampling import build_tracking_map

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--subset',type=Path,required=True);p.add_argument('--video-root',type=Path,required=True);p.add_argument('--plan-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
 p.add_argument('--sam3-root',type=Path,default=os.environ.get('SAM3_ROOT'));p.add_argument('--checkpoint',type=Path,default=os.environ.get('SAM3_CHECKPOINT'));p.add_argument('--device',type=int,default=0);p.add_argument('--persistent',action='store_true');p.add_argument('--sam-stride',type=int,default=3);p.add_argument('--target-fps',type=float);p.add_argument('--sampling-manifest-root',type=Path);p.add_argument('--boundary-radius',type=int,default=15);p.add_argument('--segment-gate-margin',type=int,default=15);p.add_argument('--offload-video-to-cpu',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--offload-state-to-cpu',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args(argv);rows=simple_rows(a.subset)
 if not rows:raise ValueError('Empty subset')
 if a.sam3_root:os.environ['SAM3_ROOT']=str(a.sam3_root.resolve())
 if a.checkpoint:os.environ['SAM3_CHECKPOINT']=str(a.checkpoint.resolve())
 tasks=[]
 for video,q in rows:
  sample=f'{video}_q{q}';plan=a.plan_root/sample/'multi_anchor_plan.json'
  if not plan.is_file():raise FileNotFoundError(plan)
  saved=read(plan)
  if (str(saved['video_id']),str(saved['question_id']))!=(video,q):raise ValueError('Saved plan identity mismatch')
  tasks.append((video,q,sample,plan))
 if a.dry_run:
  print(json.dumps({'phase':'dry_run','profile':a.profile,'queries':len(tasks),'Gemini_calls':0,'GPU_calls':0,'persistent_model':a.persistent,'dense_schedule':'timestamp10fps+exactanchors+boundarywindows' if a.profile in ('svag','vidstg') else 'legacystride+exactanchors+boundarywindows'}));return 0
 if not a.sam3_root or not a.checkpoint:raise ValueError('Specify a local --sam3-root and --checkpoint; weights are not downloaded')
 m=load_pipeline(a.profile,dense=True);configure_media(m,a.video_root);predictor=None;cache={};summary=[]
 if m.base.torch is None or not m.base.torch.cuda.is_available():raise RuntimeError('Dense SAM3 execution requires a CUDA PyTorch environment')
 m.base.torch.cuda.set_device(a.device)
 if a.persistent:predictor=m.base.build_sam3_video_predictor(gpus_to_use=[a.device])
 for video,q,sample,path in tasks:
  result_path=a.output_root/sample/'run_result.json'
  if result_path.is_file() and read(result_path).get('status')=='complete':summary.append({'sample':sample,'status':'skipped_complete'});continue
  radius=a.boundary_radius;margin=a.segment_gate_margin
  extra=[]
  if a.profile=='svag':
   media=m.base.probe_video(m.base.video_path_for(video));radius=math.ceil(float(media['fps'])*.5);margin=math.ceil(float(media['fps'])*.5);extra=['--planning-task','svag','--sam-target-fps',str(a.target_fps or 10.)]
  elif a.profile=='vidstg':
   media=m.base.probe_video(m.base.video_path_for(video));target=a.target_fps or 10.;count=int(media['num_frames']);fps=float(media['fps'])
   frozen_path=(a.sampling_manifest_root/(video+'.json')) if a.sampling_manifest_root else (a.output_root/'_sampling'/(video+'.json'))
   if not a.sampling_manifest_root:
    sparse_ids=vidstg_grid(count,fps,1.);dense_base=vidstg_grid(count,fps,target)
    save(frozen_path,{'video_id':video,'num_frames':count,'fps':fps,'sam_target_fps':target,'sampling_formula':'floor(k * source_fps / target_fps + 0.5)','sparse_manifest_sha256':object_sha(sparse_ids),'sparse_frame_indices':sparse_ids,'dense_base_frame_indices':dense_base,'dense_frame_indices':sorted(set(dense_base)|set(sparse_ids))})
   extra=['--sam-target-fps',str(target),'--sam-sampling-manifest',str(frozen_path)]
  elif a.target_fps is not None:extra=['--sam-target-fps',str(a.target_fps)]
  if a.profile in ('hcstvg','vidstg'):extra+=['--task',a.profile]
  argv2=['--video-id',video,'--question-id',q,'--reuse-plan-json',str(path),'--sam-stride',str(a.sam_stride),'--boundary-radius',str(radius),'--segment-gate-margin',str(margin),'--session-mode','joint','--propagation-start','middle','--compact-box-output','--no-render-visualizations','--no-retain-ungated-baseline','--google-cloud-project','unused-local-replay','--output-root',str(a.output_root)]+extra
  if a.offload_video_to_cpu:argv2.append('--offload-video-to-cpu')
  if a.offload_state_to_cpu:argv2.append('--offload-state-to-cpu')
  args=m.parse_args(argv2);original_sampler=m.segment_boundary_dense_frames
  if a.profile=='svag':
   saved=read(path);media=m.base.probe_video(m.base.video_path_for(video));frozen=build_tracking_map(saved,int(media['num_frames']),float(media['fps']),target_fps=a.target_fps or 10.,boundary_seconds=.5,gate_seconds=.5)
   def registered_sampler(*unused_args,**unused_kwargs):return frozen['sampled_frame_ids'],dict(frozen)
   m.segment_boundary_dense_frames=registered_sampler
  try:result=m.run(args,predictor=predictor,frames_cache=cache if a.persistent else None)
  finally:m.segment_boundary_dense_frames=original_sampler
  save(result_path,result);summary.append({'sample':sample,'status':result.get('status','complete')})
 save(a.output_root/'dense_summary.json',{'profile':a.profile,'results':summary,'Gemini_calls':0,'persistent_model':a.persistent,'output_contract':'retained legacy compact predictions plus run_result observation schedule'})
 return 0
if __name__=='__main__':raise SystemExit(main())
