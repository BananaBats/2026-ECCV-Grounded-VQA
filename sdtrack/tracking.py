"""Run one bidirectional SAM3 session for all planned identities."""
import math

import cv2
from PIL import Image

from .fusion import sane_yxyx
from .media import read_frames


def tracking_grid(plan, count):
    bounds = (plan["first_visible_frame_idx"], plan["last_visible_frame_idx"])
    endpoints = set(bounds)
    windows = {}
    for target in plan["targets"]:
        intervals = [(s["first_frame_idx"], s["last_frame_idx"])
                     for s in target["visibility_segments"]] or [bounds]
        windows[target["obj_id"]] = [(lo - 15, hi + 15) for lo, hi in intervals]
        for lo, hi in intervals:
            endpoints.update((lo, hi))
    frames = set(range(0, count, 3)) | {count - 1}
    frames.update(a["frame_idx"] for a in plan["anchors"])
    for endpoint in endpoints:
        center = min(max(0, endpoint), count - 1)
        frames.update(range(max(0, center - 15), min(count, center + 16)))
    return sorted(frames), windows


def dense_tracks(predictor, video, plan, info):
    frames, windows = tracking_grid(plan, info["num_frames"])
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
              for _, frame in read_frames(video, frames)]
    session = predictor.handle_request(request={
        "type": "start_session", "resource_path": images,
        "offload_video_to_cpu": True, "offload_state_to_cpu": False,
    })["session_id"]
    source_to_sample = {frame: i for i, frame in enumerate(frames)}
    observations = {}
    try:
        predictor.handle_request(request={"type": "reset_session", "session_id": session})
        anchor_indices = []
        for anchor in sorted(plan["anchors"], key=lambda a: a["frame_idx"]):
            index = source_to_sample[anchor["frame_idx"]]
            anchor_indices.append(index)
            for box in sorted(anchor["boxes"], key=lambda b: b["obj_id"]):
                y0, x0, y1, x1 = [v / 1000 for v in box["box_2d"]]
                reply = predictor.handle_request(request={
                    "type": "add_prompt", "session_id": session, "frame_index": index,
                    "obj_id": box["obj_id"], "points": [[x0, y0], [x1, y1]],
                    "point_labels": [2, 3], "clear_old_points": True, "rel_coordinates": True,
                })
                response = reply["outputs"]
                ids = response.get("outputs", response)["out_obj_ids"]
                if box["obj_id"] not in ids:
                    raise RuntimeError(f"SAM3 did not register object {box['obj_id']}")

        last = len(frames) - 1
        candidates = [i for i in anchor_indices if i < last] or anchor_indices
        start = min(candidates, key=lambda index: abs(index - last / 2))
        stream = predictor.handle_stream_request(request={
            "type": "propagate_in_video", "session_id": session,
            "start_frame_index": start, "propagation_direction": "both",
        })
        for item in stream:
            frame = frames[int(item["frame_index"])]
            output = item["outputs"]
            boxes = {}
            for oid, xywh in zip(output["out_obj_ids"], output["out_boxes_xywh"], strict=True):
                oid = int(oid)
                if not any(lo <= frame <= hi for lo, hi in windows[oid]):
                    continue
                x, y, w, h = map(float, xywh)
                if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
                    continue
                try:
                    boxes[oid] = sane_yxyx([y * 1000, x * 1000, (y + h) * 1000, (x + w) * 1000])
                except ValueError:
                    continue
            observations[frame] = boxes
    finally:
        predictor.handle_request(request={"type": "close_session", "session_id": session})
    tracks = {}
    for frame, targets in observations.items():
        for identity, box in targets.items():
            tracks.setdefault(identity, {})[frame] = box
    return tracks
