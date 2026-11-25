"""Image-only sparse localization with retained benchmark prompts and validators.

--prepare-only writes exact JPEGs and independent call descriptors, without API
calls. Run normally for Standard API, or use pipeline.sparse_batch for Batch.
Perception/HC retain their original stride30 grid; SVAG/VidSTG use native1Hz
plus the saved-anchor coverage exception. VidSTG's recorded visible profile is
kept separate from HC/Perception/SVAG amodal localization.
"""
from __future__ import annotations
import argparse,inspect,json,os,subprocess
from pathlib import Path
from .common import PROFILES,read,save,digest,object_sha,load_sparse,generation_client,simple_rows,profile_config
from .sampling_sparse import native_grid,coverage_schedule
from .sampling_vidstg import sample_frames as vidstg_grid

def response_class(module,profile):
 if profile!='svag':return module.SparseResponse
 from pydantic import Field,create_model
 obj=create_model('SVAGSparseObject',__base__=module.SparseObject,object_id=(int,Field(ge=0)))
 frame=create_model('SVAGSparseFrame',__base__=module.SparseFrame,objects=(list[obj],Field()))
 return create_model('SVAGSparseResponse',__base__=module.SparseResponse,frames=(list[frame],Field(min_length=1)))

def make_prompt(module,profile,plan,frames,video,mode,stride):
 if profile=='vidstg' and mode=='visible' and stride==0:
  return module.build_visible_prompt(plan,frames,video,0).replace('These are fixed 0-frame-grid positions.','These are exact original frame IDs sampled at 1 FPS on the native video clock.')
 if 'box_mode' in inspect.signature(module.build_prompt).parameters:result=module.build_prompt(plan,frames,video,mode,stride)
 else:result=module.build_prompt(plan,frames,video)
 return result.replace('fixed 30-frame-grid','FPS-aware source-frame-grid') if profile=='svag' else result

def extract(video,frames,directory):
 directory.mkdir(parents=True,exist_ok=True)
 if all((directory/f'frame_{f:06d}.jpg').is_file() for f in frames):return
 expression='+'.join(f'eq(n\\,{f})' for f in frames)
 command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-i',str(video),'-map','0:v:0','-an','-threads','1','-filter_threads','1','-vf',f'select={expression},settb=1/1,setpts=N','-r','1','-q:v','2','-start_number','0','-y',str(directory/'selected_%06d.jpg')]
 subprocess.run(command,check=True,capture_output=True,timeout=3600);files=sorted(directory.glob('selected_*.jpg'))
 if len(files)!=len(frames):raise ValueError('Exact source frame extraction count mismatch')
 for f,path in zip(frames,files):path.replace(directory/f'frame_{f:06d}.jpg')

def export_sample(root,sample,module,Response):
 root=Path(root);d=root/sample;meta=read(d/'manifest.json');plan=read(meta['plan_path']);allframes=[]
 for item in meta['requests']:
  path=Path(item['call_path'])
  if not path.is_file() or read(path).get('status')!='complete':return False
  record=read(path)
  if record.get('request_sha256')!=object_sha(item):raise ValueError('Cached response does not match prepared request')
  parsed=Response.model_validate(record['response']);module.validate_response(parsed,plan,item['frame_indices']);allframes.extend(parsed.frames)
 predictions={str(f):[] for f in meta['frame_indices']}
 for frame in allframes:
  for item in frame.objects:
   if meta['box_mode']=='visible' and item.visibility=='fully_occluded':continue
   y0,x0,y1,x1=item.box_2d;predictions[str(frame.frame_idx)].append({'object_id':item.object_id,'visibility':item.visibility,'confidence':item.confidence,'box_2d_yxyx_1000':item.box_2d,'xyxy_normalized':[x0/1000,y0/1000,x1/1000,y1/1000]})
 save(d/'sparse_predictions.json',{'video_id':meta['video_id'],'question_id':meta['question_id'],'answer_noun_phrase':plan.get('answer_noun_phrase',''),'targets':plan['targets'],'frame_stride':meta['frame_stride'],'predictions_by_frame':predictions})
 save(d/'run_result.json',{'status':'complete','sample':sample,'profile':meta['profile'],'box_mode':meta['box_mode'],'num_sparse_frames':len(predictions),'num_boxes':sum(map(len,predictions.values())),'GT_inputs':[]});return True

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--subset',type=Path,required=True);p.add_argument('--plan-root',type=Path,required=True);p.add_argument('--video-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
 p.add_argument('--project',default=os.environ.get('GOOGLE_CLOUD_PROJECT'));p.add_argument('--location',default='global');p.add_argument('--model',help='override the recorded sparse model for this profile');p.add_argument('--frames-per-call',type=int,default=8);p.add_argument('--frame-stride',type=int,default=30);p.add_argument('--sampling-policy',choices=['profile','stride'],default='profile');p.add_argument('--box-mode',choices=['amodal','visible']);p.add_argument('--grounding-margin',type=int);p.add_argument('--thinking-level',default='high');p.add_argument('--max-output-tokens',type=int,default=32768);p.add_argument('--prepare-only',action='store_true');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args(argv);rows=simple_rows(a.subset)
 settings = profile_config(a.profile)
 a.model = a.model or settings['sparse_model']
 if a.grounding_margin is None:
  a.grounding_margin = settings.get('sparse_grounding_margin', 30)
 if not 1<=a.frames_per_call<=15 or a.frame_stride<1:raise ValueError('Invalid sparse cadence or batch size')
 if a.dry_run:print(json.dumps({'phase':'dry_run','profile':a.profile,'questions':len(rows),'API_calls':0,'cadence':'native1Hz+coverageanchors' if a.profile=='svag' else ('native1Hz' if a.profile=='vidstg' and a.sampling_policy=='profile' else 'source_stride'+str(a.frame_stride))}));return 0
 module=load_sparse(a.profile);Response=response_class(module,a.profile);mode=a.box_mode or settings['sparse_box_mode'];client=None;descriptors=[]
 if a.profile not in ('hcstvg','vidstg') and mode!='amodal':raise ValueError('This profile retains an amodal-only prompt')
 for video,q in rows:
  sample=f'{video}_q{q}';planpath=(a.plan_root/sample/'multi_anchor_plan.json').resolve();plan=read(planpath);vp=a.video_root/f'{video}.mp4'
  if not vp.exists():
   matches=sorted(a.video_root.glob(video+'.*'))
   if len(matches)!=1:raise FileNotFoundError(vp)
   vp=matches[0]
  info=module.probe_video(vp);count=info.get('metadata_frame_count')
  if not count:
   from .common import load_pipeline
   count=load_pipeline(a.profile).base.probe_video(vp)['num_frames'];info['metadata_frame_count']=count
  if a.profile=='svag':frames,additions,reasons,warnings=coverage_schedule(plan,native_grid(count,info['fps']));batch=min(a.frames_per_call,max(1,64//len(plan['targets'])))
  elif a.profile=='vidstg' and a.sampling_policy=='profile':
   frames=vidstg_grid(count,info['fps'],1.);batch=a.frames_per_call;additions=[]
  else:
   frames=list(range(0,count,a.frame_stride));batch=a.frames_per_call;additions=[]
   if a.profile in ('hcstvg','vidstg'):frames=module.grounding_window(plan,frames,a.grounding_margin)
  if not frames:raise ValueError(f'No sparse frames remain for {sample}')
  directory=(a.output_root/sample).resolve();frame_dir=directory/'frames';extract(vp,frames,frame_dir);calls=[]
  for index,start in enumerate(range(0,len(frames),batch)):
   part=frames[start:start+batch];prompt=make_prompt(module,a.profile,plan,part,info,mode,0 if a.profile=='vidstg' and a.sampling_policy=='profile' else a.frame_stride)
   request={'key':f'{sample}__call_{index:03d}','sample':sample,'profile':a.profile,'model':a.model,'box_mode':mode,'frame_indices':part,'frame_dir':str(frame_dir),'plan_path':str(planpath),'plan_sha256':digest(planpath),'prompt':prompt,'generation':{'temperature':0.,'thinking_level':a.thinking_level,'max_output_tokens':a.max_output_tokens},'images':[{'frame':f,'path':str(frame_dir/f'frame_{f:06d}.jpg'),'sha256':digest(frame_dir/f'frame_{f:06d}.jpg')} for f in part],'call_path':str(directory/'calls'/f'call_{index:03d}.json')}
   calls.append(request);descriptors.append(request);(directory/'calls').mkdir(exist_ok=True);(directory/'calls'/f'call_{index:03d}.prompt.txt').write_text(prompt+'\n')
  manifest={'sample':sample,'video_id':video,'question_id':q,'profile':a.profile,'box_mode':mode,'plan_path':str(planpath),'video_path':str(vp.resolve()),'source_sha256':digest(vp),'frame_indices':frames,'frame_stride':None if a.profile=='svag' or (a.profile=='vidstg' and a.sampling_policy=='profile') else a.frame_stride,'coverage_anchor_frames':additions,'frames_per_call':batch,'requests':calls,'GT_inputs':[]}
  previous=directory/'manifest.json'
  if previous.exists() and read(previous)!=manifest:raise ValueError('Prepared sparse inputs changed; use a new output root')
  save(previous,manifest)
  if a.prepare_only:continue
  if client is None:client=generation_client(a.project,a.location)
  from google.genai import types
  for request in calls:
   cp=Path(request['call_path']);key=object_sha(request)
   if cp.exists():
    record=read(cp)
    if record.get('request_sha256')!=key:raise ValueError('Existing call inputs differ')
    if record.get('status')=='complete':continue
    raise RuntimeError('Call has an unresolved or Batch-owned response; inspect it before resubmission')
   contents=module.build_contents(types,request['prompt'],request['frame_indices'],frame_dir);generation=request['generation'];config=types.GenerateContentConfig(temperature=0.,thinking_config=types.ThinkingConfig(thinking_level=generation['thinking_level']),max_output_tokens=generation['max_output_tokens'],response_mime_type='application/json',response_schema=Response)
   save(cp,{'status':'sent','request_sha256':key,'transport':'standard'})
   response=client.models.generate_content(model=a.model,contents=contents,config=config)
   raw=response.parsed if response.parsed is not None else json.loads(response.text);save(cp,{'status':'response_received','request_sha256':key,'transport':'standard','response':raw.model_dump(mode='json') if hasattr(raw,'model_dump') else raw,'usage_metadata':response.usage_metadata.model_dump(mode='json') if response.usage_metadata else None})
   parsed=Response.model_validate(raw);module.validate_response(parsed,plan,request['frame_indices']);save(cp,{'status':'complete','request_sha256':key,'transport':'standard','response':parsed.model_dump(mode='json'),'usage_metadata':response.usage_metadata.model_dump(mode='json') if response.usage_metadata else None})
  export_sample(a.output_root,sample,module,Response)
 save(a.output_root/'prepared_requests.json',{'profile':a.profile,'requests':descriptors,'prepared_source_revision':'portable-adapter-original-sparse-prompts','GT_inputs':[]})
 print(json.dumps({'phase':'prepared' if a.prepare_only else 'complete','requests':len(descriptors),'questions':len(rows),'API_calls':0 if a.prepare_only else 'see call journals'}));return 0
if __name__=='__main__':raise SystemExit(main())
