"""Dependency-light configuration and retained-source loading helpers."""
from __future__ import annotations
import hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
PROFILES=('perception','svag','hcstvg','vidstg')
LEGACY=Path(__file__).resolve().parent/'_legacy'
FORBIDDEN={'gt','ground_truth','answer','answers','reference_answer','answer_track_ids','object_tracking','bounding_boxes','gt_tracks','spatial','tracks','action_intervals'}

def read(path):return json.loads(Path(path).read_text())

def profile_config(profile):
    return read(Path(__file__).resolve().parents[1] / 'configs' / f'{profile}.json')

def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def queries(path):
 value=read(path);rows=value['queries'] if isinstance(value,dict) and 'queries' in value else value
 if not isinstance(rows,list):raise ValueError('Queries must be a list, or {"queries": [...]} with question-only records')
 result=[];seen=set()
 for row in rows:
  if not isinstance(row,dict) or any(str(k).lower() in FORBIDDEN for k in row):raise ValueError('Ground truth/reference answers are forbidden in inference queries')
  v=str(row['video_id']);q=str(row['question_id']);text=str(row['question']).strip();sample=f'{v}_q{q}'
  if not text or Path(v).name!=v or Path(sample).name!=sample or sample in seen:raise ValueError('Require safe, unique video/question IDs and nonempty question text')
  seen.add(sample);result.append({**row,'video_id':v,'question_id':q,'question':text,'sample':sample})
 return result

def load_pipeline(profile,*,dense=False):
 group=('svag_dense' if dense else 'svag_plan') if profile=='svag' else ('vidstg_dense' if dense and profile=='vidstg' else profile)
 directory=LEGACY/group;os.environ['V6_ROOT']=str(directory)
 for key in ['box_baseline','hcstvg_prompts']:
  prior=sys.modules.get(key)
  if prior is not None and Path(prior.__file__).parent!=directory:del sys.modules[key]
 sys.path.insert(0,str(directory))
 return load_file(directory/'multi_anchor_box_baseline.py',f'_retained_{group}_pipeline')
def load_sparse(profile):
 directory=LEGACY/f'{profile}_sparse';sys.path.insert(0,str(directory));sys.modules.pop('sparse_common',None)
 return load_file(directory/'generate_sparse.py',f'_retained_{profile}_sparse')
def load_file(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def configure_media(module,video_root):
 module.base.VIDEO_ROOT=Path(video_root).resolve()

def task_kwargs(function,profile):
 params=inspect.signature(function).parameters
 if profile=='svag' and 'planning_task' in params:return {'planning_task':'svag'}
 if 'task' in params:return {'task':profile if profile in ('hcstvg','vidstg') else 'perception_test'}
 return {}

def generation_client(project,location='global'):
 if not project:raise ValueError('Set --project or GOOGLE_CLOUD_PROJECT to your own project')
 from google import genai
 from google.genai import types
 return genai.Client(vertexai=True,project=project,location=location,http_options=types.HttpOptions(api_version='v1',timeout=300000))

def simple_rows(subset):
 rows=[]
 for line in Path(subset).read_text().splitlines():
  if line.strip():
   video,q=line.split('\t');rows.append((video.strip(),q.strip()))
 if len(rows)!=len(set(rows)):raise ValueError('Subset contains duplicates')
 return rows
