"""Portable, annotation-free replay of saved dense/sparse fusion experiments.

Use ``--profile minimal`` for the one-threshold method or ``legacy_v13`` for
historical reliability-gated results. These are different methods, not aliases.
No model, API, annotation, or GPU dependency is imported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from . import core
from .controls import counterfactual_correction, fixed_support
from .legacy import fuse_dense_sparse as legacy
from .legacy.sparse_common import load_json,save_json,load_subset,sample_name

MODES=('full','no_expansion_fixed_support','no_residual_fixed_support',
       'no_residual_sam_priority','wo_residual','wo_residual_controlled','wo_gap',
       'dense_only','sparse_only','dense_sampled_linear')


def linear_all(track):
    """Every interior integer frame; no extrapolation or endpoint IoU gate."""
    output=dict(track)
    for a,b in zip(sorted(track),sorted(track)[1:]):
        for frame in range(a+1,b):
            w=(frame-a)/(b-a)
            output[frame]=[track[a][k]*(1-w)+track[b][k]*w for k in range(4)]
    return output


def fuse_tracks(sam_tracks,sparse_tracks,*,residual_tau=.3,gap_tau=.3,mode='full'):
    """Return YXYX/1000 tracks and per-target provenance, without reading GT.

    Sparse IDs define the output identities, exactly as the production method.
    ``wo_residual`` is the literal HC/Vid control: Gemini owns sparse frames.
    ``wo_residual_controlled`` keeps Full's coordinates at sparse frames.
    The positive-extent and no-residual fixed-support controls hold Full's
    output frame set and interpolation brackets fixed (Perception ablation).
    """
    if mode not in MODES:raise ValueError('Unknown mode')
    r=core.validate_threshold(residual_tau);g=core.validate_threshold(gap_tau)
    if mode=='dense_only':return {i:dict(t) for i,t in sam_tracks.items()},[]
    if mode=='dense_sampled_linear':return {i:linear_all(t) for i,t in sam_tracks.items()},[]
    if mode=='sparse_only':return {i:linear_all(t) for i,t in sparse_tracks.items()},[]
    output={};details=[]
    for oid,sp in sorted(sparse_tracks.items()):
        sam=sam_tracks.get(oid,{})
        corrected,anchors=core.correct_sam_track(sam,sp,r)
        merged=dict(sp);merged.update(corrected)
        full=core.interpolate_stable_gaps(merged,g)
        if mode=='full':track=full
        elif mode=='wo_gap':track=merged
        elif mode=='wo_residual':
            raw=dict(sam);raw.update(sp)
            track=core.interpolate_stable_gaps(raw,g)
        elif mode=='wo_residual_controlled':
            raw=dict(sam)
            for f in sp:raw[f]=list(merged[f])
            track=core.interpolate_stable_gaps(raw,g)
        elif mode=='no_residual_sam_priority':
            raw=dict(sp);raw.update(sam)
            track=core.interpolate_stable_gaps(raw,g)
        elif mode=='no_residual_fixed_support':
            raw=dict(sp);raw.update(sam)
            track=fixed_support(raw,full)
        else:
            nosize=counterfactual_correction(sam,sp,anchors)
            raw=dict(sp);raw.update(nosize)
            track=fixed_support(raw,full)
        if track:output[oid]=track
        details.append({'object_id':oid,'residual_anchor_frames':anchors,
            'num_sam_frames':len(sam),'num_sparse_frames':len(sp),
            'num_output_frames':len(track),'raw_sam_without_residual':bool(sam) and not anchors})
    return output,details


def load_dense(path,video,q,policy='legacy_min1'):
    if policy=='legacy_min1':return legacy.load_dense_tracks(path,video,q)
    result={}
    for row in load_json(path)[video]['grounded_question'][str(q)]:
        oid=int(row['id']);fs=list(map(int,row['frame_ids']));boxes=row['bounding_boxes']
        if oid in result or fs!=sorted(set(fs)) or len(fs)!=len(boxes):raise ValueError('Invalid dense ID/frame binding')
        result[oid]={}
        for f,b in zip(fs,boxes):
            x0,y0,x1,y1=map(float,b)
            if not all(math.isfinite(v) and 0<=v<=1 for v in [x0,y0,x1,y1]) or x0>=x1 or y0>=y1:raise ValueError('Invalid dense box')
            result[oid][f]=[1000*y0,1000*x0,1000*y1,1000*x1]
    return result


def payload(video,q,tracks,allow_small=False):
    if not allow_small:return legacy.official_payload(video,q,tracks)
    rows=[]
    def value(x):
        if not math.isfinite(x) or not -1e-9<=x<=1000+1e-9:raise ValueError('Fusion box outside normalized domain')
        return min(1000.,max(0.,x))/1000.
    for oid,tr in sorted(tracks.items()):
        fs=sorted(tr)
        rows.append({'id':oid,'score':1.,'frame_ids':fs,'bounding_boxes':[
            [value(tr[f][1]),value(tr[f][0]),value(tr[f][3]),value(tr[f][2])] for f in fs]})
    return {video:{'grounded_question':{str(q):rows}}}


def validate_subset(path):
    rows=load_subset(path)
    if not rows or len(set(rows))!=len(rows):raise ValueError('Subset must contain unique nonempty video/question pairs')
    for v,q in rows:
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]+',x) or x in ('.','..') for x in (v,q)):raise ValueError('Unsafe video/question path component')
    return rows


def run(args):
    rows=validate_subset(args.subset);fallback=set(load_subset(args.fallback_samples)) if args.fallback_samples else set()
    if not fallback<=set(rows):raise ValueError('Fallback list is outside selected cohort')
    if args.profile=='legacy_v13' and args.mode!='full':raise ValueError('Ablations are defined for minimal profile; legacy_v13 must use mode=full')
    if args.profile=='legacy_v13' and args.dense_box_policy!='legacy_min1':raise ValueError('legacy_v13 preserves its original input box validity rule')
    out=Path(args.output_root)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory is not empty; earlier experiments are preserved')
    out.mkdir(parents=True,exist_ok=True)
    merged={};summaries=[];failures=[]
    for video,q in rows:
        name=sample_name(video,q)
        try:
            dd=Path(args.dense_root)/name;dp=dd/'predictions.json';pp=Path(args.plan_root)/name/'multi_anchor_plan.json'
            full=load_dense(dp,video,q,args.dense_box_policy);plan=load_json(pp)
            nf=legacy.infer_num_frames(dd,full)
            if args.mode=='dense_only' or (video,q) in fallback:
                result=load_json(dp);details=[];recovery={'source':'unmodified_dense_export','fallback':(video,q) in fallback}
            elif args.profile=='legacy_v13':
                result,meta=legacy.fuse_sample(SimpleNamespace(plan_root=Path(args.plan_root),dense_root=Path(args.dense_root),sparse_root=Path(args.sparse_root),min_anchor_iou=args.min_anchor_iou,min_shared=args.min_shared,gate_iou=args.gate_iou,short_gap=args.short_gap),video,q)
                details=meta['targets'];recovery=meta['sam_coverage_recovery']
            else:
                sam,recovery=legacy.recover_sampled_sam_tracks(dd,full)
                sp=legacy.load_sparse_tracks(Path(args.sparse_root)/name/'sparse_predictions.json')
                legacy.direct_id_mapping(core.plan_target_ids(plan),set(sam),set(sp))
                tracks,details=fuse_tracks(sam,sp,residual_tau=args.residual_tau,gap_tau=args.gap_tau,mode=args.mode)
                result=payload(video,q,tracks,allow_small=args.dense_box_policy=='minimal_valid')
            legacy.validate_official_sample(result,video,q,nf)
            meta={'sample':name,'profile':args.profile,'mode':args.mode,'residual_tau':args.residual_tau,'gap_tau':args.gap_tau,'sam_coverage_recovery':recovery,'targets':details,'num_video_frames':nf,'fixed_dense_fallback':(video,q) in fallback,'annotation_inputs':[]}
            if args.profile=='legacy_v13':meta['legacy_thresholds']={'min_anchor_iou':args.min_anchor_iou,'min_shared':args.min_shared,'gate_iou':args.gate_iou,'style_agreement_iou':.3,'nearest_residual_frames':60,'short_gap':args.short_gap,'long_gap_bracket_iou':.3}
            save_json(out/name/'predictions.json',result);save_json(out/name/'fusion_meta.json',meta)
            if not args.no_merged:legacy.merge_sample(merged,result)
            summaries.append(meta);print('complete',name,flush=True)
        except Exception as e:
            failures.append({'sample':name,'error_type':type(e).__name__,'error':str(e)});print('FAILED',name,type(e).__name__,str(e),flush=True)
    summary={'profile':args.profile,'mode':args.mode,'num_selected':len(rows),'num_complete':len(summaries),'num_failed':len(failures),'model_calls':0,'GPU_calls':0,'annotation_inputs':[],'residual_tau':args.residual_tau,'gap_tau':args.gap_tau,'dense_box_policy':args.dense_box_policy,'samples':summaries,'failures':failures}
    save_json(out/'fusion_summary.json',summary)
    if not failures and not args.no_merged:save_json(out/'predictions.json',merged)
    return int(bool(failures))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['subset','plan-root','dense-root','sparse-root','output-root']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--profile',choices=['minimal','legacy_v13'],default='minimal')
    p.add_argument('--mode',choices=MODES,default='full')
    p.add_argument('--residual-tau',type=float,default=.3);p.add_argument('--gap-tau',type=float,default=.3)
    p.add_argument('--dense-box-policy',choices=['legacy_min1','minimal_valid'],default='legacy_min1',help='VidSTG frozen replay uses minimal_valid, retaining small positive boxes')
    p.add_argument('--fallback-samples',type=Path,help='Explicit fixed dense-only fallback TSV, shared by all arms (e.g. stored VidSTG failures)')
    p.add_argument('--no-merged',action='store_true',help='Write per-sample files only to limit RAM use')
    p.add_argument('--min-anchor-iou',type=float,default=.25);p.add_argument('--min-shared',type=int,default=3);p.add_argument('--gate-iou',type=float,default=.3);p.add_argument('--short-gap',type=int,default=6)
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    a.residual_tau=core.validate_threshold(a.residual_tau);a.gap_tau=core.validate_threshold(a.gap_tau)
    return run(a)

if __name__=='__main__':raise SystemExit(main())
