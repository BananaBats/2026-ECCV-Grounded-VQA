#!/usr/bin/env python3
"""Prepare HC-STVG input captions and separate evaluation-only GT.

Raw annotations/videos are never modified. Videos remain in their native MP4/MKV
containers and are symlinked, without transcoding, into isolated split folders.
"""
import argparse
import concurrent.futures
import hashlib
import json
import random
from pathlib import Path

SPLITS = {
    'hcstvg_v1_test': ('dataset/test.json','dataset/videos',1160),
    'hcstvg_v2_val': ('dataset/v2/val.json','dataset/v2/videos',2000),
}


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def convert_annotation(name,a,video):
    """Frame 1 in source annotations is decoded frame 0; xywh -> xyxy."""
    caption=str(a.get('caption',a.get('English',''))).strip()
    if not caption:raise ValueError('Missing caption: '+name)
    aw,ah=(a['width'],a['height']) if 'width' in a else (a['img_size'][1],a['img_size'][0])
    if (aw,ah)!=(video['width'],video['height']):
        raise ValueError(f'Annotation/video size mismatch for {name}: {(aw,ah)} vs {video}')
    first=int(a['st_frame'])-1;last=first+len(a['bbox'])
    if not (0<=first<last):raise ValueError('Invalid annotation interval: '+name)
    boxes={};degenerate=[]
    for i,b in enumerate(a['bbox']):
        x,y,w,h=map(float,b)
        frame=first+i
        if w<=0 or h<=0:degenerate.append(frame)
        if frame<video['num_frames']:boxes[str(frame)]=[x,y,x+w,y+h]
    clipped_last=min(last,video['num_frames'])
    if first>=clipped_last:raise ValueError('GT lies outside the input video: '+name)
    stem=Path(name).stem
    question={'grounded_question':[{'id':'c0','question':caption,'sentence_type':'declarative'}]}
    gt={'clip_id':stem,'question_id':'c0','vid':stem,'sentence_type':'declarative',
        'target_type':'person','target_tid':0,'eval_frames':[0,video['num_frames']],
        'tube':[first,clipped_last],'boxes_xyxy':boxes,'width':aw,'height':ah,
        'degenerate_source_gt_frames':degenerate,
        'source_annotation':{'st_frame_1_based':a['st_frame'],'bbox_count':len(a['bbox']),
                             'st_time':a['st_time'],'ed_time':a['ed_time'],'img_num':a['img_num']}}
    return question,gt


def prepare(root,split,probe_headers=True,workers=8):
    relann,relvideos,expected=SPLITS[split]
    source=root/relann;raw=json.loads(source.read_text())
    assert len(raw)==expected,(split,len(raw),expected)
    out=root/'dataset'/split
    out.mkdir(exist_ok=True,parents=True)
    (out/'videos').mkdir(exist_ok=True)
    def inspect(item):
        name,a=item;src=root/relvideos/name
        if not src.is_file():raise FileNotFoundError(src)
        aw,ah=(a['width'],a['height']) if 'width' in a else (a['img_size'][1],a['img_size'][0])
        n,fps=a['img_num'],a['img_num']/20.
        if probe_headers:
            import cv2
            cap=cv2.VideoCapture(str(src))
            try:
                if not cap.isOpened():raise RuntimeError('Cannot open '+str(src))
                n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
                actual=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                if actual!=(aw,ah):raise ValueError(f'Size mismatch: {name}, annotation={(aw,ah)}, video={actual}')
            finally:cap.release()
        if n<=0 or fps<=0:raise ValueError('Invalid media metadata: '+name)
        stem=Path(name).stem;link=out/'videos'/name
        if link.is_symlink():
            if link.resolve()!=src.resolve():raise RuntimeError('Existing link points elsewhere: '+str(link))
        elif link.exists():raise RuntimeError('Refusing to replace existing video: '+str(link))
        else:link.symlink_to(src.resolve())
        video={'vid':stem,'begin_fid':0,'end_fid':n,'num_frames':n,'fps':fps,'width':aw,'height':ah,
               'video_path':str(link),'source_video_path':str(src),'native_container':src.suffix,
               'frame_count_source':'container_header' if probe_headers else 'annotation_fallback'}
        question,gt=convert_annotation(name,a,video)
        return stem,question,gt,video,int(a['img_num'])!=n
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        records=list(pool.map(inspect,sorted(raw.items())))
    assert len({r[0] for r in records})==expected
    annotations={r[0]:r[1] for r in records}
    gt={r[0]+'_qc0':r[2] for r in records}
    clips={r[0]:r[3] for r in records}
    save(out/'annotations'/f'grounded_question_{split}.json',annotations)
    save(out/'eval'/'hcstvg_gt.json',gt)
    save(out/'hcstvg_clips.json',clips)
    lines=[f'{r[0]}\tc0' for r in records]
    (out/'subset.tsv').write_text('\n'.join(lines)+'\n')
    pilot=random.Random(20261107).sample(lines,min(10,len(lines)))
    (out/'subset_pilot10.tsv').write_text('\n'.join(pilot)+'\n')
    report={'split':split,'samples':expected,'source_annotation':str(source),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'native_extensions':{ext:sum(r[3]['native_container']==ext for r in records) for ext in ['.mp4','.mkv']},
            'header_count_differs_from_annotation':sum(r[4] for r in records),
            'degenerate_source_gt_boxes':sum(len(r[2]['degenerate_source_gt_frames']) for r in records),
            'samples_with_degenerate_source_gt':sum(bool(r[2]['degenerate_source_gt_frames']) for r in records),
            'input_contains_only_caption_and_id':True,'ground_truth_only_in_eval':True,
            'frame_convention':'source st_frame is 1-based, internal half-open [st_frame-1,st_frame-1+len(bbox))',
            'gt_coordinates':'original xywh -> xyxy; no spatial clipping or amodal/visible relabeling',
            'frame_count_note':'Headers may overcount MKV frames. Planning uses HCSTVG_STRICT_FRAME_COUNT=1 to count decodable frames without GT.',
            'model_calls':0}
    save(out/'preparation_report.json',report)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--split',choices=list(SPLITS)+['all'],default='all')
    ap.add_argument('--annotation-metadata-only',action='store_true')
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    for split in SPLITS if args.split=='all' else [args.split]:
        print(json.dumps(prepare(args.root.resolve(),split,not args.annotation_metadata_only,args.workers),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
