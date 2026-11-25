"""Four-call-family planner using retained benchmark-specific experimental sources.

Planning stops after exact-frame boxes. It never starts SAM or reads GT.
The public adapter follows the original headless worker sequence; account,
cluster scheduling, and directory registration are replaced by explicit paths.
"""
from __future__ import annotations
import argparse,inspect,json,math,os
from pathlib import Path
from .common import PROFILES,queries,load_pipeline,configure_media,task_kwargs,generation_client,read,save,digest,object_sha,profile_config
from .svag_fps_sampling import sample_anchor_frames

class Journal:
 def __init__(self,module,directory):self.original=module.base.call_function;self.directory=directory;self.index=0;self.source_sha256=[digest(Path(module.__file__)),digest(Path(module.base.__file__))]
 def call(self,client,model,name,schema,parts,*args,**kwargs):
  self.index+=1;p=self.directory/f'call_{self.index:03d}.json'
  descriptions=[]
  for part in parts:
   inline=getattr(part,'inline_data',None)
   descriptions.append({'text':getattr(part,'text',None),'mime':getattr(inline,'mime_type',None),'video_metadata':getattr(part,'video_metadata',None).model_dump(mode='json') if getattr(part,'video_metadata',None) is not None else None,'bytes_sha256':__import__('hashlib').sha256(inline.data).hexdigest() if inline is not None and inline.data else None})
  key=object_sha({'source_sha256':self.source_sha256,'model':model,'name':name,'schema':schema,'parts':descriptions,'args':args,'kwargs':kwargs})
  if p.exists():
   old=read(p)
   if old['request_sha256']!=key:raise ValueError('Cached call inputs changed; use a new output directory')
   if old['status']!='complete':raise RuntimeError('A previous request has no validated response. Inspect the journal before retrying paid inference.')
   return old['arguments'],old['metadata']
  save(p,{'status':'sent','request_sha256':key,'function':name,'model':model})
  arguments,metadata=self.original(client,model,name,schema,parts,*args,**kwargs)
  save(p,{'status':'complete','request_sha256':key,'function':name,'model':model,'arguments':arguments,'metadata':metadata})
  return arguments,metadata

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--video-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
 p.add_argument('--model',help='override the recorded planner model for this profile');p.add_argument('--project',default=os.environ.get('GOOGLE_CLOUD_PROJECT'));p.add_argument('--location',default='global');p.add_argument('--sample',action='append');p.add_argument('--max-anchor-frames',type=int,default=3)
 p.add_argument('--anchor-stride',type=int,default=10);p.add_argument('--anchor-target-fps',type=float,default=3.);p.add_argument('--video-fps',type=float,default=4.);p.add_argument('--proxy-max-frames',type=int,default=192);p.add_argument('--proxy-max-dimension',type=int,default=512);p.add_argument('--boundary-radius',type=int,default=15);p.add_argument('--dry-run',action='store_true')
 a=p.parse_args(argv);rows=queries(a.queries);rows=[r for r in rows if not a.sample or r['sample'] in a.sample]
 a.model = a.model or profile_config(a.profile)['planner_model']
 if not rows:raise ValueError('No selected query')
 if a.max_anchor_frames<1 or a.anchor_stride<1:raise ValueError('Positive anchor budget and stride required')
 if a.profile=='hcstvg':os.environ['HCSTVG_BOX_MODE']='amodal';os.environ['HCSTVG_STRICT_FRAME_COUNT']='1'
 m=load_pipeline(a.profile);configure_media(m,a.video_root);client=None;done=[]
 for row in rows:
  v,q=row['video_id'],row['question_id'];video=Path(row['video_path']).resolve() if row.get('video_path') else m.base.video_path_for(v)
  meta=m.base.probe_video(video);count=int(meta['num_frames']);fps=float(meta['fps']);directory=a.output_root/row['sample'];path=directory/'multi_anchor_plan.json'
  allowed=sample_anchor_frames(count,fps,a.anchor_target_fps,5.) if a.profile=='svag' else m.base.stride_frames(count,a.anchor_stride)
  signature=object_sha({'query':row,'profile':a.profile,'model':a.model,'video_sha256':digest(video),'source_sha256':digest(Path(m.__file__)),'base_sha256':digest(Path(m.base.__file__)),'allowed_frames':allowed,'max_anchor_frames':a.max_anchor_frames,'video_fps':a.video_fps,'proxy_max_frames':a.proxy_max_frames,'proxy_max_dimension':a.proxy_max_dimension})
  if path.exists():
   if read(directory/'run_result.json').get('input_signature')!=signature:raise ValueError('Existing plan inputs differ; choose a fresh output root')
   done.append({'sample':row['sample'],'status':'skipped_complete'});continue
  if a.dry_run:done.append({'sample':row['sample'],'status':'dry_run','profile':a.profile,'allowed_anchor_frames':allowed,'model_calls':0});continue
  if client is None:client=generation_client(a.project,a.location)
  proxy,proxy_meta=m.base.build_planning_proxy_streaming(video,directory,a.proxy_max_frames,a.proxy_max_dimension)
  journal=Journal(m,directory/'calls');original=m.base.call_function;m.base.call_function=journal.call
  try:
   noun,noun_meta=m.request_noun_phrase(client,a.model,proxy,row['question'],a.video_fps,**task_kwargs(m.request_noun_phrase,a.profile))
   phrase,reason=noun.answer_noun_phrase,noun.reasoning
   critic,critic_meta=m.verify_noun_phrase(client,a.model,proxy,row['question'],phrase,a.video_fps,**task_kwargs(m.verify_noun_phrase,a.profile))
   if critic.get('verdict')=='wrong' and str(critic.get('corrected_noun_phrase','')).strip():phrase=critic['corrected_noun_phrase'].strip();reason=f"[critic override] {critic.get('reasoning','')}"
   count_hint=critic.get('expected_instance_count');minimum=1 if a.profile=='svag' else 2;hint=count_hint if isinstance(count_hint,int) and count_hint>=minimum else None
   extra=task_kwargs(m.request_anchor_selection,a.profile)
   if 'num_frames' in inspect.signature(m.request_anchor_selection).parameters:extra['num_frames']=count
   selection,selection_meta=m.request_anchor_selection(client,a.model,proxy,row['question'],phrase,allowed,a.max_anchor_frames,a.video_fps,instance_count_hint=hint,**extra)
   if a.profile=='svag':m.validate_svag_selection(selection,count)
   if a.profile in ('hcstvg','vidstg') and hasattr(m,'clamp_grounding_segment'):selection=m.clamp_grounding_segment(selection,count)
   boundary=math.ceil(fps*.5) if a.profile=='svag' else a.boundary_radius
   sample_kwargs={'base_indices':allowed} if a.profile=='svag' else {}
   sampled,sampling=m.segment_boundary_dense_frames(count,a.anchor_stride,selection,boundary,**sample_kwargs)
   identities={t.obj_id:t for t in selection.targets};anchors=[]
   for anchor in selection.anchors:
    frame=m.base.load_video_frames_at(video,[anchor.frame_idx])[anchor.frame_idx];exact=m.base.save_rgb(frame,directory/'anchor_frames'/f'frame_{anchor.frame_idx:06d}.png')
    boxes,box_meta=m.request_anchor_boxes(client,a.model,proxy,exact,row['question'],phrase,anchor,[identities[i] for i in anchor.obj_ids],a.video_fps,**task_kwargs(m.request_anchor_boxes,a.profile))
    anchors.append(m.SavedAnchor(frame_idx=anchor.frame_idx,sam_frame_idx=sampled.index(anchor.frame_idx),obj_ids=anchor.obj_ids,reason=anchor.reason,boxes=boxes.boxes,exact_frame_path=str(Path(exact).resolve()),call_metadata=box_meta))
   values=dict(video_id=v,question_id=q,question=row['question'],answer_noun_phrase=phrase,noun_reasoning=reason,noun_confidence=noun.confidence,decision_summary=selection.decision_summary,first_visible_frame_idx=selection.first_visible_frame_idx,last_visible_frame_idx=selection.last_visible_frame_idx,visibility_boundary_reason=selection.visibility_boundary_reason,targets=selection.targets,anchors=anchors,sam_stride=a.anchor_stride,boundary_radius=boundary,max_anchor_frames=a.max_anchor_frames,anchor_policy='gemini',model=a.model,gemini_video_fps=a.video_fps,call_metadata={'noun_phrase':{**noun_meta,'noun_verification':{**critic,'call_metadata':critic_meta}},'anchor_selection':selection_meta,'gemini_planning_proxy':proxy_meta,'release_profile':a.profile,'input_signature':signature})
   if a.profile=='svag':values.update(planning_task='svag',planner_revision='svag-v3',planner_provider='gemini',planner_input_format='labeled_video_proxy',planner_frame_limit=a.proxy_max_frames)
   if a.profile in ('hcstvg','vidstg'):values.update(task=a.profile,grounding_segment=getattr(selection,'grounding_segment',None))
   plan=m.SavedMultiAnchorPlan(**values);save(path,plan.model_dump(mode='json'));save(directory/'sam_frame_map.json',{**sampling,'num_original_frames':count,'source_fps':fps,'sam_to_original_frame':{str(i):f for i,f in enumerate(sampled)}})
   result={'sample':row['sample'],'status':'complete','input_signature':signature,'targets':len(plan.targets),'anchors':len(plan.anchors),'logical_model_calls':journal.index,'GT_in_model_inputs':False,'SAM_started':False};save(directory/'run_result.json',result);done.append(result)
  finally:m.base.call_function=original
 print(json.dumps({'results':done,'dry_run':a.dry_run},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
