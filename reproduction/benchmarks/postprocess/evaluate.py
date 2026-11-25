"""Evaluation adapters for explicit benchmark frame protocols.

Inference and fusion do not import this module. Ground truth enters here only.
Perception reports question-macro HOTA; STVG reports sequence-combined HOTA.
All machine-readable metrics are fractions; *_points fields multiply by 100.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .legacy.sparse_common import load_json,save_json,load_subset,sample_name
from .metrics import perception_eval,stvg_eval,stvg_hota


def paper_frames(gt,clip):
    """Exact stored TubeDETR 5fps / max200 selection, original frame IDs."""
    begin,end=map(int,gt['eval_frames']);rate=5./float(clip['fps']);frames=[begin]
    for f in range(begin,end):
        if int(frames[-1]*rate)<int(f*rate):frames.append(f)
    if len(frames)>200:frames=[frames[(j*len(frames))//200] for j in range(200)]
    return frames


def sampled_sequence_score(gt,prediction,frames):
    """Stored VidSTG scorer on an explicit evaluation grid, no GT filling."""
    gboxes={str(f):b for f,b in gt['boxes_xyxy'].items() if gt['tube'][0]<=int(f)<gt['tube'][1]}
    prediction=prediction or {'boxes_xyxy':{},'pred_segment':None};pboxes=prediction['boxes_xyxy']
    gids,pids,sim=stvg_eval.sequence_data(gboxes,pboxes,frames)
    result=stvg_hota.eval_sequence(gids,pids,sim)
    segment=prediction['pred_segment'];lo,hi=map(int,gt['tube'])
    if segment is None:tiou=viou=0.
    else:
        p0,p1=map(int,segment);inter=max(0,min(hi,p1)-max(lo,p0));union=(hi-lo)+(p1-p0)-inter
        tiou=inter/union if union else 0.
        union_frames=[f for f in frames if min(lo,p0)<=f<max(hi,p1)]
        numerator=sum(stvg_eval.single_iou(gboxes[str(f)],pboxes[str(f)]) for f in frames
            if max(lo,p0)<=f<min(hi,p1) and str(f) in gboxes and str(f) in pboxes)
        viou=numerator/max(len(union_frames),1)
    return result,{'tIoU':float(tiou),'vIoU':float(viou),'HOTA':float(np.mean(result['HOTA']))}


def perception(predictions,annotations,subset):
    rows=perception_eval.evaluate(predictions,annotations,subset)
    metrics={k:float(np.mean([r[k] for r in rows])) for k in ['HOTA','DetA','AssA','LocA']}
    return {'benchmark':'Perception Test','aggregation':'question_macro','questions':len(rows),
        'scoring_frames':'union of answer GT frames after initial_tracking_box; predictions filtered to this grid',
        'metrics_fraction':metrics,'metrics_points':{k:100*v for k,v in metrics.items()},'per_question':rows}


def reference_statistics(gt,prediction,queried,positive):
    lo,hi=map(int,gt['eval_frames']);a,b=map(int,gt['tube']);truth={int(f):box for f,box in gt['boxes_xyxy'].items() if lo<=int(f)<hi and a<=int(f)<b}
    pred=(prediction or {}).get('boxes_xyxy',{});masks={'all_GT':set(truth),'queried_reference_GT':set(truth)&queried,'non_queried_reference_GT':set(truth)-queried,'positive_reference_GT':set(truth)&positive,'non_positive_reference_GT':set(truth)-positive};out={}
    for name,frames in masks.items():
        found=[f for f in frames if str(f) in pred];total=sum(stvg_eval.single_iou(truth[f],pred[str(f)]) for f in found)
        out[name]={'GT_frames':len(frames),'predicted_boxes_on_GT':len(found),'IoU_sum':total,
            'mean_IoU_missing_as_zero':total/len(frames) if frames else None,
            'mean_IoU_on_present_predictions':total/len(found) if found else None}
    return out


def stvg(gt,tubes,*,protocol='hc_native',subset=None,clips=None,reference_root=None):
    if protocol not in ['hc_native','vidstg_native','vidstg_5fps_max200']:raise ValueError('Unknown frame protocol')
    tubes=tubes.get('samples',tubes);selected={sample_name(*p) for p in subset} if subset else set(gt)
    if not selected<=set(gt):raise ValueError('Selected samples absent from evaluation GT')
    if not selected:raise ValueError('Empty selected cohort')
    if protocol=='vidstg_5fps_max200' and clips is None:raise ValueError('5fps/max200 requires clip fps metadata')
    result={};scalars={};missing=[];overlap={}
    for sample in sorted(selected):
        pred=tubes.get(sample)
        if pred is None:missing.append(sample)
        truth=gt[sample]
        if protocol=='hc_native':r,s=stvg_eval.evaluate_sample(truth,pred,stride=1,oracle_temporal=False)
        else:
            v,_=sample.rsplit('_q',1)
            frames=paper_frames(truth,clips[v]) if protocol=='vidstg_5fps_max200' else list(range(*truth['eval_frames']))
            r,s=sampled_sequence_score(truth,pred,frames)
        result[sample]=(r,s);scalars[sample]=s
        if reference_root is not None:
            if clips is None:raise ValueError('Reference statistics require clip begin_fid metadata')
            folder=Path(reference_root)/sample;manifest=load_json(folder/'manifest.json');sp=load_json(folder/'sparse_predictions.json');v,_=sample.rsplit('_q',1);offset=int(clips[v]['begin_fid'])
            queried={int(f)+offset for f in manifest['frame_indices']}
            positive={int(f)+offset for f,items in sp['predictions_by_frame'].items() if items}
            overlap[sample]=reference_statistics(truth,pred,queried,positive)
    metrics=stvg_eval.aggregate(result);numeric={k:v for k,v in metrics.items() if k!='num_sentences'}
    summary={'benchmark':'STVG','protocol':protocol,'oracle_temporal':False,'aggregation':'TrackEval sequence_combination; m_vIoU/m_tIoU query_macro',
        'metrics_fraction':metrics,'metrics_points':{k:100*v for k,v in numeric.items()},
        'questions':len(selected),'missing_predictions':missing,'per_question':scalars}
    if overlap:
        grouped={}
        for k in next(iter(overlap.values())):
            n=sum(r[k]['GT_frames'] for r in overlap.values());found=sum(r[k]['predicted_boxes_on_GT'] for r in overlap.values());total=sum(r[k]['IoU_sum'] for r in overlap.values())
            grouped[k]={'GT_frames':n,'predicted_boxes_on_GT':found,'IoU_sum':total,'micro_mean_IoU_missing_as_zero':total/n if n else None,'micro_mean_IoU_on_present_predictions':total/found if found else None}
        summary['nonreference_analysis']={'per_question':overlap,'combined':grouped,'GT_reference_overlap_fraction':grouped['positive_reference_GT']['GT_frames']/max(1,grouped['all_GT']['GT_frames'])}
    return summary


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='export-stvg':
        from .legacy.stvg_export import main as export
        return export(argv[1:])
    if argv and argv[0]=='svagzip':
        from .submission import main as submission
        return submission(argv[1:])
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='benchmark',required=True)
    a=sub.add_parser('perception',help='Question-macro official notebook-compatible HOTA')
    for n in ['predictions','annotations','subset','output']:a.add_argument('--'+n,type=Path,required=True)
    a=sub.add_parser('stvg',help='STVG tube metrics under an explicit frame protocol')
    for n in ['gt','tubes','output']:a.add_argument('--'+n,type=Path,required=True)
    a.add_argument('--subset',type=Path);a.add_argument('--clips',type=Path);a.add_argument('--reference-root',type=Path)
    a.add_argument('--protocol',choices=['hc_native','vidstg_native','vidstg_5fps_max200'],required=True)
    sub.add_parser('export-stvg',help='Use export-stvg --help for legacy-compatible tube export')
    sub.add_parser('svagzip',help='Use svagzip --help for official test ZIP serialization')
    args=p.parse_args(argv)
    if args.benchmark=='perception':summary=perception(load_json(args.predictions),load_json(args.annotations),load_subset(args.subset))
    elif args.benchmark=='stvg':summary=stvg(load_json(args.gt),load_json(args.tubes),protocol=args.protocol,subset=load_subset(args.subset) if args.subset else None,clips=load_json(args.clips) if args.clips else None,reference_root=args.reference_root)
    else:raise ValueError('Choose an adapter command')
    save_json(args.output,summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['per_question','nonreference_analysis']},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
