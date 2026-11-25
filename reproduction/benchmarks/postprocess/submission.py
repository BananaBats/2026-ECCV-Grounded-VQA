"""Serialize public SVAG test predictions for Codabench, without GT or upload.

Each public-metadata record includes local_video_id, question_id, dataset,
query_id, query, video_id (organizer integer), video_name, video_length,
width and height. Predictions are normalized XYXY; spatial output is pixel
XYWH at zero-based original frame indices. Plan actions remain inclusive on
input and become [first,last+1,score] windows in the official schema.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from .legacy.sparse_common import load_json,save_json,sample_name
from .svag_schema import make_query,validate_paired_queries,temporal_fingerprint
DATASETS=('OVIS','MOT17','MOT20')


def write_zip(destination,records,loader,plans):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix(destination.suffix+'.partial')
    if tmp.exists():raise FileExistsError(tmp)
    meta=[]
    try:
        with ZipFile(tmp,'w',compression=ZIP_DEFLATED) as archive:
            with archive.open('submission.json','w',force_zip64=True) as f:
                emit=lambda s:f.write(s.encode('utf-8'))
                encode=lambda v:json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                emit('{"datasets":[')
                for d,ds in enumerate(DATASETS):
                    if d:emit(',')
                    emit('{"name":'+encode(ds)+',"queries":[')
                    selected=[r for r in records if r['dataset']==ds]
                    for j,r in enumerate(selected):
                        v,q=r['local_video_id'],str(r['question_id']);plan=load_json(plans/sample_name(v,q)/'multi_anchor_plan.json');pred=loader(v,q)
                        tracks=pred[v]['grounded_question'][q]  # missing query is an incomplete run, not an empty prediction
                        query=make_query(r,plan,tracks,{'width':int(r['width']),'height':int(r['height']),'num_frames':int(r['video_length'])})
                        if j:emit(',')
                        emit(encode(query));meta.append({'dataset':ds,'query_id':r['query_id'],'temporal_fingerprint':temporal_fingerprint(plan,r['video_length'])})
                    emit(']}')
                emit(']}\n')
        os.replace(tmp,destination)
    except Exception:
        tmp.unlink(missing_ok=True);raise
    return meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--plan-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--predictions',type=Path);g.add_argument('--prediction-root',type=Path)
    p.add_argument('--allow-partial',action='store_true',help='For local schema tests only; full organizer test metadata is required for actual submission')
    a=p.parse_args(argv);records=load_json(a.records);records=records.get('records',records) if isinstance(records,dict) else records
    keys=[(r['dataset'],r['query_id']) for r in records]
    if not records or len(keys)!=len(set(keys)):raise ValueError('Public metadata query keys must be unique and nonempty')
    if any(r['dataset'] not in DATASETS for r in records):raise ValueError('Unknown SVAG dataset')
    if not a.allow_partial and set(r['dataset'] for r in records)!=set(DATASETS):raise ValueError('Actual test submission requires all three datasets')
    merged=load_json(a.predictions) if a.predictions else None
    loader=(lambda v,q:merged) if merged is not None else (lambda v,q:load_json(a.prediction_root/sample_name(v,q)/'predictions.json'))
    audit=write_zip(a.output,records,loader,a.plan_root)
    save_json(a.output.with_suffix(a.output.suffix+'.audit.json'),{'queries':len(records),'frame_unit_temporal':True,'action_based_spatial_filtering':False,'interpolation_added':False,'server_upload':False,'partial_schema_test':a.allow_partial,'temporal_fingerprints':audit})
    print('Wrote',a.output,'with',len(records),'queries; no server upload performed');return 0

if __name__=='__main__':raise SystemExit(main())
