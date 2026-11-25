"""Export only query text/IDs from prepared benchmark annotations.

This is an explicit allow-list conversion: answers, target tracks, action times,
GT boxes, and annotation metadata are never copied into the inference manifest.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
ALLOWED=('video_id','question_id','question','sample','video_path')

def prepare(annotations,subset,video_root):
    seen=set();rows=[]
    for video,q in subset:
        if (video,q) in seen:raise ValueError('Duplicate selected question')
        seen.add((video,q))
        if Path(video).name!=video:raise ValueError('Video ID is not a basename')
        entry=annotations[video]
        questions=entry['grounded_question']
        question=next(x for x in questions if str(x['id'])==str(q))
        text=str(question['question']).strip()
        if not text:raise ValueError('Empty question')
        files=[Path(video_root)/(video+s) for s in ['.mp4','.mkv','.avi','.mov']]
        available=[p for p in files if p.is_file()]
        if len(available)!=1:raise ValueError(f'{video}: expected exactly one video container, found {len(available)}')
        rows.append({'video_id':video,'question_id':str(q),'question':text,'sample':f'{video}_q{q}','video_path':str(available[0].resolve())})
    return {'input_contract':'Question text and IDs only; no reference answers or target annotations.','queries':rows}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for a in ['annotations','subset','video-root','output']:p.add_argument('--'+a,type=Path,required=True)
    a=p.parse_args(argv);annotations=json.loads(a.annotations.read_text());subset=[tuple(line.split('\t')) for line in a.subset.read_text().splitlines() if line.strip()]
    if not subset:raise ValueError('Empty cohort')
    result=prepare(annotations,subset,a.video_root)
    if a.output.exists():raise FileExistsError('Query manifest already exists; choose a new path')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'queries':len(result['queries']),'reference_answer_fields':0,'API_calls':0,'output':str(a.output)}));return 0
if __name__=='__main__':raise SystemExit(main())
