"""Convert frozen SVAG predictions to the organizer's SVAGEval query schema.

This adapter changes serialization only. Spatial observations stay at their
zero-based original frame positions. Saved inclusive action intervals [a,b]
become frame-unit temporal windows [a,b+1,1.0], independently of spatial boxes.
No annotations, model calls, interpolation or action-based spatial gating occur.
"""
from __future__ import annotations

import hashlib
import json
import math


REVISION = 'svag-official-submission-common-plan-actions-v1'
DATASETS = {'OVIS', 'MOT17', 'MOT20'}
QUERY_HEADER_FIELDS = ('query_id','query','video_id','video_name','video_length')
QUERY_FIELDS = set(QUERY_HEADER_FIELDS)|{'tracks'}


def _number(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _validate_record(record):
    if record.get('dataset') not in DATASETS:
        raise ValueError('Unknown official SVAG dataset')
    for field in ['query_id','video_id']:
        if not _nonnegative_int(record.get(field)):
            raise ValueError(f'Official {field} must be a nonnegative integer')
    if type(record.get('video_length')) is not int or record['video_length'] < 1:
        raise ValueError('Official video_length must be a positive frame count')
    for field in ['query','video_name']:
        if not isinstance(record.get(field),str) or not record[field]:
            raise ValueError(f'Official {field} must be a nonempty string')


def plan_temporal_tracks(plan,num_frames):
    """Return every saved plan ID's action windows without merging or sorting."""
    if type(num_frames) is not int or num_frames < 1:
        raise ValueError('num_frames must be a positive original frame count')
    targets=plan.get('targets')
    if not isinstance(targets,list) or not targets:
        raise ValueError('Saved plan must contain target identities')
    result={}
    for target in targets:
        if not isinstance(target,dict) or not _nonnegative_int(target.get('obj_id')):
            raise ValueError('Saved plan obj_id must be a nonnegative integer')
        identity=target['obj_id']
        if identity in result:
            raise ValueError('Saved plan has duplicate obj_id values')
        segments=target.get('action_segments')
        if not isinstance(segments,list) or not segments:
            raise ValueError(f'Saved plan target {identity} has no action_segments')
        windows=[]
        for segment in segments:
            if not isinstance(segment,dict):
                raise ValueError('Saved action segment must be an object')
            a,b=segment.get('first_frame_idx'),segment.get('last_frame_idx')
            if not (_nonnegative_int(a) and _nonnegative_int(b) and a <= b < num_frames):
                raise ValueError('Saved action segment is outside original frame bounds')
            windows.append([a,b+1,1.0])
        result[identity]=windows
    return result


def temporal_fingerprint(plan,num_frames):
    """Bind the complete saved plan's temporal policy, including absent tracks."""
    temporal=plan_temporal_tracks(plan,num_frames)
    payload={'video_length':num_frames,'temporal_by_plan_id':[
        {'track_id':identity,'temporal':windows} for identity,windows in sorted(temporal.items())]}
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_query(query,*,allowed_ids=None,expected_record=None,video_meta=None):
    """Validate one organizer-format query and return counts.

    ``expected_record`` binds dataset-independent query metadata; dataset identity
    remains in the enclosing submission dataset. Query IDs need not be globally
    unique across OVIS, MOT17 and MOT20. Optional dimensions check pixel bounds.
    """
    if not isinstance(query,dict) or set(query) != QUERY_FIELDS:
        raise ValueError('Official query has missing or unexpected schema fields')
    record={key:query[key] for key in QUERY_HEADER_FIELDS}
    record['dataset']=expected_record['dataset'] if expected_record is not None else 'OVIS'
    _validate_record(record)
    if expected_record is not None:
        _validate_record(expected_record)
        if any(query[field] != expected_record[field] for field in QUERY_HEADER_FIELDS):
            raise ValueError('Official query metadata differs from its registered record')
    allowed=None if allowed_ids is None else set(allowed_ids)
    if allowed is not None and any(not _nonnegative_int(identity) for identity in allowed):
        raise ValueError('Allowed plan IDs must be nonnegative integers')
    tracks=query['tracks']
    if not isinstance(tracks,list):
        raise ValueError('Official tracks must be an array')
    width=height=None
    if video_meta is not None:
        width,height=video_meta.get('width'),video_meta.get('height')
        if type(width) is not int or type(height) is not int or min(width,height) < 1:
            raise ValueError('Video width and height must be positive integers')
        if 'num_frames' in video_meta and (type(video_meta['num_frames']) is not int
                or video_meta['num_frames'] != query['video_length']):
            raise ValueError('Video dimensions metadata has a different frame count')
    ids,boxes,windows=set(),0,0
    for track in tracks:
        if not isinstance(track,dict) or set(track) != {'track_id','spatial','temporal'}:
            raise ValueError('Official track has missing or unexpected schema fields')
        identity=track['track_id']
        if (not _nonnegative_int(identity) or identity in ids
                or (allowed is not None and identity not in allowed)):
            raise ValueError('Official track IDs are duplicate or absent from the saved plan')
        ids.add(identity)
        spatial=track['spatial']
        if not isinstance(spatial,list) or len(spatial) != query['video_length']:
            raise ValueError('Official spatial array must have video_length entries')
        for box in spatial:
            if box is None:continue
            if (not isinstance(box,list) or len(box) != 4 or not all(_number(value) for value in box)
                    or min(box[:2]) < 0 or min(box[2:]) <= 0):
                raise ValueError('Official spatial box must be finite positive pixel XYWH')
            # Multiplication and subtraction of valid normalized coordinates may
            # differ by a floating-point ULP at the full image boundary.
            if width is not None and (box[0]+box[2] > width+1e-9 or box[1]+box[3] > height+1e-9):
                raise ValueError('Official spatial box exceeds the original image bounds')
            boxes+=1
        temporal=track['temporal']
        if not isinstance(temporal,list) or not temporal:
            raise ValueError('Official track needs saved action temporal windows')
        for window in temporal:
            if (not isinstance(window,list) or len(window) != 3
                    or not _nonnegative_int(window[0]) or type(window[1]) is not int
                    or not window[0] < window[1] <= query['video_length']
                    or not _number(window[2]) or not 0 <= window[2] <= 1):
                raise ValueError('Official temporal window must be frame-unit [a,b+1,score]')
            windows+=1
    return {'num_tracks':len(tracks),'num_boxes':boxes,'num_temporal_windows':windows}


def make_query(record,plan,tracks,video_meta):
    """Serialize normalized XYXY tracks using immutable saved plan actions."""
    _validate_record(record)
    width,height=video_meta.get('width'),video_meta.get('height')
    if type(width) is not int or type(height) is not int or min(width,height) < 1:
        raise ValueError('Video width and height must be positive integers')
    if (type(video_meta.get('num_frames',record['video_length'])) is not int
            or video_meta.get('num_frames',record['video_length']) != record['video_length']):
        raise ValueError('Video frame count differs from its registered release record')
    if 'question_id' in plan and str(plan['question_id']) != str(record['query_id']):
        raise ValueError('Saved plan question ID differs from official release record')
    if 'question' in plan and plan['question'] != record['query']:
        raise ValueError('Saved plan query text differs from official release record')
    temporal=plan_temporal_tracks(plan,record['video_length'])
    if not isinstance(tracks,list):
        raise ValueError('Internal tracks must be an array')
    result=[];ids=set()
    for track in tracks:
        if not isinstance(track,dict):raise ValueError('Internal track must be an object')
        identity=track.get('id')
        if not _nonnegative_int(identity) or identity in ids or identity not in temporal:
            raise ValueError('Internal track IDs are duplicate or absent from the saved plan')
        ids.add(identity)
        frames,boxes=track.get('frame_ids'),track.get('bounding_boxes')
        if (not isinstance(frames,list) or not isinstance(boxes,list)
                or not frames or len(frames)!=len(boxes)):
            raise ValueError('Internal frame/box arrays must be nonempty and equal length')
        if (any(not _nonnegative_int(frame) or frame >= record['video_length'] for frame in frames)
                or frames != sorted(set(frames))):
            raise ValueError('Internal frames must be unique sorted original video indices')
        spatial=[None]*record['video_length']
        for frame,box in zip(frames,boxes):
            if (not isinstance(box,list) or len(box)!=4 or not all(_number(value) for value in box)
                    or not 0<=box[0]<box[2]<=1 or not 0<=box[1]<box[3]<=1):
                raise ValueError('Internal spatial box must be finite positive normalized XYXY')
            x0,y0,x1,y1=box
            spatial[frame]=[x0*width,y0*height,(x1-x0)*width,(y1-y0)*height]
        result.append({'track_id':identity,'spatial':spatial,
                       'temporal':[list(window) for window in temporal[identity]]})
    query={field:record[field] for field in QUERY_HEADER_FIELDS}
    query['tracks']=result
    validate_query(query,allowed_ids=temporal,expected_record=record,video_meta=video_meta)
    return query


def validate_paired_queries(dense,fused,plan,num_frames):
    """Verify both spatial methods use the same immutable action predictions."""
    temporal=plan_temporal_tracks(plan,num_frames)
    for query in [dense,fused]:
        validate_query(query,allowed_ids=temporal)
        if query['video_length'] != num_frames:
            raise ValueError('Paired export differs from the registered frame count')
        for track in query['tracks']:
            if track['temporal'] != temporal[track['track_id']]:
                raise ValueError('Paired export temporal prediction differs from the saved plan')
    if any(dense[field]!=fused[field] for field in QUERY_HEADER_FIELDS):
        raise ValueError('Paired official query identities differ')
    dense_ids={track['track_id'] for track in dense['tracks']}
    fused_ids={track['track_id'] for track in fused['tracks']}
    return {'passed':True,'same_plan_action_verified':True,
            'temporal_fingerprint':temporal_fingerprint(plan,num_frames),
            'common_track_ids':sorted(dense_ids&fused_ids),
            'dense_only_track_ids':sorted(dense_ids-fused_ids),
            'fused_only_track_ids':sorted(fused_ids-dense_ids)}
