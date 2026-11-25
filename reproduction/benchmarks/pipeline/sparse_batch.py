"""Vertex Batch transport for prepared, independent sparse image requests.

This adapter reuses the actual transport class and SDK wire-schema serializer.
It changes orchestration only; prompts, original JPEGs, generation settings and
response validation come from pipeline.sparse. Dense Direct chunks are dependent
and must not be submitted through this sparse-only Batch entrypoint.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from .common import read,save,digest,object_sha,load_sparse
from .sparse import response_class,export_sample

def canonical_contents(value):
 request=value.get('request',value);result=[]
 for content in request['contents']:
  parts=[]
  for p in content['parts']:
   if 'text' in p:parts.append({'text':p['text']})
   elif p.get('fileData') or p.get('file_data'):
    f=p.get('fileData',p.get('file_data'));parts.append({'fileUri':f.get('fileUri',f.get('file_uri')),'mimeType':f.get('mimeType',f.get('mime_type'))})
  result.append({'role':content.get('role','user'),'parts':parts})
 return object_sha(result)

def wire_generation(Response,settings):
 # This is the same offline official-SDK serialization used by the original
 # sparse Batch experiment, including Pydantic $defs resolution.
 from google import genai
 from google.genai import _common,models,types
 from google.auth.credentials import AnonymousCredentials
 client=genai.Client(vertexai=True,project='offline-schema-conversion',location='global',credentials=AnonymousCredentials())
 try:
  c=types.GenerateContentConfig(temperature=0.,thinking_config=types.ThinkingConfig(thinking_level=settings['thinking_level']),max_output_tokens=settings['max_output_tokens'],response_mime_type='application/json',response_schema=Response)
  wire=models._GenerateContentConfig_to_vertex(client._api_client,c)
  return _common.encode_unserializable_types(_common.convert_to_dict(wire))
 finally:client.close()

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','submit','status','fetch']);p.add_argument('--requests',type=Path,required=True);p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--project',default=os.environ.get('GOOGLE_CLOUD_PROJECT'));p.add_argument('--bucket',required=True);p.add_argument('--prefix',default='sd-track-sparse');p.add_argument('--location',default='global');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args(argv);prepared=read(a.requests);requests=prepared['requests'];profile=prepared['profile'];root=a.requests.resolve().parent
 if not requests:raise ValueError('No prepared sparse requests')
 if a.dry_run:print(json.dumps({'phase':'dry_run','action':a.action,'requests':len(requests),'profile':profile,'API_calls':0}));return 0
 if not a.project:raise ValueError('Supply your own --project')
 import google.auth
 from ._legacy.batch_cloud import BatchCloud
 credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform']);cloud=BatchCloud(credentials,a.project,a.location,a.bucket)
 a.batch_dir.mkdir(parents=True,exist_ok=True);stagepath=a.batch_dir/'stage.json';jobpath=a.batch_dir/'job.json';module=load_sparse(profile);Response=response_class(module,profile)
 try:
  if a.action=='stage':
   if stagepath.exists():
    if read(stagepath)['prepared_requests_sha256']!=digest(a.requests):raise ValueError('Prepared requests changed')
    print(json.dumps(read(stagepath)));return 0
   cloud.ensure_bucket();rows=[];mapping={};images={};candidates=[]
   for item in requests:
    cp=Path(item['call_path'])
    if cp.exists():
     r=read(cp)
     if r.get('request_sha256')!=object_sha(item):raise ValueError('Call request changed')
     if r.get('status')=='complete':continue
     raise RuntimeError('Call already exists without a complete validated response; reconcile its owner first')
    parts=[{'text':item['prompt']}]
    for image in item['images']:
     path=Path(image['path'])
     if digest(path)!=image['sha256']:raise ValueError('Prepared JPEG changed')
     key=image['sha256']
     if key not in images:images[key]=cloud.upload_file(path,f'{a.prefix}/images/{key}.jpg','image/jpeg')['uri']
     parts.append({'fileData':{'fileUri':images[key],'mimeType':'image/jpeg'}})
    generation=wire_generation(Response,item['generation'])
    if profile=='vidstg':
     from .batch_schema import inline_json_schema
     schema=inline_json_schema(Response.model_json_schema());frame_array=schema['properties']['frames'];group=item['frame_indices'];frame_array.update(minItems=len(group),maxItems=len(group));frame_array['items']['properties']['frame_idx'].update(minimum=min(group),maximum=max(group));objects=frame_array['items']['properties']['objects'];known=sorted(int(t['obj_id']) for t in read(item['plan_path'])['targets']);objects['maxItems']=min(8,len(known));objects['items']['properties']['object_id'].update(minimum=min(known),maximum=max(known));generation={'temperature':0.,'thinkingConfig':{'thinkingLevel':item['generation']['thinking_level'].upper()},'maxOutputTokens':item['generation']['max_output_tokens'],'responseMimeType':'application/json','responseJsonSchema':schema}
    row={'request':{'contents':[{'role':'user','parts':parts}],'generationConfig':generation}};key=canonical_contents(row)
    if key in mapping:raise ValueError('Duplicate echoed contents in one Batch; split these duplicates into separate jobs to keep association unambiguous')
    mapping[key]=item;rows.append(row);candidates.append(item)
   if not rows:raise ValueError('All prepared sparse calls already completed')
   path=a.batch_dir/'requests.jsonl';path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows));uploaded=cloud.upload_file(path,f'{a.prefix}/requests/{digest(path)}.jsonl','application/jsonl')
   stage={'prepared_requests_sha256':digest(a.requests),'profile':profile,'model':candidates[0]['model'],'input_uri':uploaded['uri'],'output_prefix':f'gs://{a.bucket}/{a.prefix}/outputs/{digest(path)[:16]}/','display_name':'sd-track-sparse-'+digest(path)[:16],'rows':mapping,'GT_inputs':[]};save(stagepath,stage);print(json.dumps({'phase':'staged','requests':len(rows),'stage':str(stagepath),'model_calls':0}));return 0
  stage=read(stagepath)
  if stage['prepared_requests_sha256']!=digest(a.requests):raise ValueError('Prepared request descriptor changed')
  if a.action=='submit':
   if jobpath.exists():
    existing=read(jobpath)
    if existing.get('job',{}).get('name'):print(json.dumps(existing));return 0
    raise RuntimeError('Unresolved submit intent: reconcile by display_name before any second create call')
   save(jobpath,{'phase':'submit_intent','display_name':stage['display_name'],'stage_sha256':digest(stagepath)})
   for item in stage['rows'].values():save(item['call_path'],{'status':'batch_reserved','request_sha256':object_sha(item),'transport':'vertex_batch','display_name':stage['display_name']})
   job=cloud.submit(stage['input_uri'],stage['output_prefix'],stage['display_name'],model=stage['model']);save(jobpath,{'phase':'submitted','job':job,'stage_sha256':digest(stagepath)});print(json.dumps(job));return 0
  record=read(jobpath);job=record.get('job',{})
  if not job.get('name'):raise RuntimeError('Submission response unknown; reconcile the saved display_name with cloud.list_jobs_by_display_name before proceeding')
  current=cloud.get_job(job['name']);save(a.batch_dir/'status.json',current)
  if a.action=='status':print(json.dumps(current));return 0
  state=str(current.get('state','')).split('.')[-1]
  if state not in {'JOB_STATE_SUCCEEDED','JOB_STATE_PARTIALLY_SUCCEEDED'}:raise RuntimeError(f'Batch is not ready to import: {state}')
  consumed=set();failures=[]
  for n,obj in enumerate(cloud.list_output(stage['output_prefix'])):
   if not obj['name'].endswith('.jsonl'):continue
   path=a.batch_dir/'downloaded'/f'output_{n:04d}.jsonl';cloud.download_object(obj['uri'],path)
   for line in path.read_text().splitlines():
    result=json.loads(line);key=canonical_contents(result)
    if key not in stage['rows'] or key in consumed:raise ValueError('Unknown or duplicate response contents')
    consumed.add(key);item=stage['rows'][key];cp=Path(item['call_path']);response=result.get('response',{})
    try:
     if result.get('status'):raise ValueError(str(result['status']))
     candidate=response['candidates'][0];finish=str(candidate.get('finishReason',candidate.get('finish_reason','')))
     if any(x in finish for x in ['SAFETY','PROHIBITED_CONTENT','BLOCKLIST','SPII','RECITATION']):raise ValueError('Model blocked/refused this call')
     text=''.join(p.get('text','') for p in candidate['content']['parts'] if not p.get('thought'));payload=json.loads(text);parsed=Response.model_validate(payload);module.validate_response(parsed,read(item['plan_path']),item['frame_indices'])
    except (KeyError,IndexError,ValueError,TypeError) as error:
     failures.append({'key':item['key'],'error':str(error)});save(cp,{'status':'validation_failed','request_sha256':object_sha(item),'transport':'vertex_batch','raw_output':result});continue
    save(cp,{'status':'complete','request_sha256':object_sha(item),'transport':'vertex_batch','response':parsed.model_dump(mode='json'),'usage_metadata':response.get('usageMetadata',response.get('usage_metadata'))})
  samples=sorted({item['sample'] for item in requests});completed=[s for s in samples if export_sample(root,s,module,Response)]
  summary={'phase':'import_complete','received':len(consumed),'expected':len(stage['rows']),'missing_contents':sorted(set(stage['rows'])-consumed),'completed_samples':completed,'failures':failures};save(a.batch_dir/'import_summary.json',summary);print(json.dumps(summary));return 1 if failures or summary['missing_contents'] else 0
 finally:cloud.close()
if __name__=='__main__':raise SystemExit(main())
