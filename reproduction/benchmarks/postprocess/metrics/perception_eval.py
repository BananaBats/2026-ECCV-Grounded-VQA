#!/usr/bin/env python3
"""Per-question HOTA for Perception Test grounded VQA predictions.

Mirrors ``run_iou`` + ``eval_sequences`` from the official notebook
(third_party/perception_test/baselines/grounded_vqa.ipynb):
* a GT track starts at the one-hot ``initial_tracking_box`` index;
* predictions are filtered to annotated GT frame ids (the notebook's
  ``filter_pred_boxes``), boxes clipped to [0, 1], IoU on xyxy;
* at most 10 tracks per question (highest ``score`` first, if present);
* a question's score is the mean over the 19 alphas, the split score is the
  mean over questions.
The HOTA input is keyed by frame id, so partial-coverage predictions are
aligned correctly instead of positionally.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from .perception_hota import eval_sequence  # noqa: E402

RELEASE_ROOT = HERE.parents[1]
DEFAULT_ANNOTATIONS = RELEASE_ROOT / "dataset/valid/annotations/grounded_question_valid.json"
DEFAULT_SUBSET = RELEASE_ROOT / "config/val200.tsv"
COLUMNS = ("sample", "video_id", "question_id", "HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr",
           "AssRe", "AssPr", "num_gt_ids", "num_tracker_ids", "num_gt_dets", "num_tracker_dets")


def iou_xyxy(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = gt.reshape(-1, 4)
    pred = pred.reshape(-1, 4)
    x0 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y0 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x1 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y1 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = area_gt[:, None] + area_pred[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def gt_track_start(track: dict[str, Any]) -> int:
    flags = np.asarray(track.get("initial_tracking_box") or [], dtype=int)
    if flags.size == 0 or np.count_nonzero(flags) == 0:
        return 0
    return int(np.nonzero(flags)[0][0])


def question_data(gt_tracks: list[dict[str, Any]], answers: list[int],
                  pred_tracks: list[dict[str, Any]], max_num_tracks: int = 10) -> dict[str, Any]:
    by_id = {int(t["id"]): t for t in gt_tracks}
    gt_boxes: list[dict[int, np.ndarray]] = []
    for answer in answers:
        track = by_id[int(answer)]
        start = gt_track_start(track)
        frames = [int(f) for f in track["frame_ids"][start:]]
        boxes = np.asarray(track["bounding_boxes"][start:], dtype=float).reshape(-1, 4)
        gt_boxes.append(dict(zip(frames, boxes)))
    gt_frames = sorted({f for track in gt_boxes for f in track})

    if len(pred_tracks) > max_num_tracks:
        pred_tracks = sorted(pred_tracks, key=lambda t: float(t.get("score", 0.0)), reverse=True)[:max_num_tracks]
    pred_boxes: list[dict[int, np.ndarray]] = []
    gt_frame_set = set(gt_frames)
    for track in pred_tracks:
        frames = [int(f) for f in track["frame_ids"]]
        boxes = np.asarray(track["bounding_boxes"], dtype=float).reshape(-1, 4)
        boxes = np.minimum(np.maximum(boxes, 0.0), 1.0)
        pred_boxes.append({f: b for f, b in zip(frames, boxes) if f in gt_frame_set})

    gt_ids, tracker_ids, sims = [], [], []
    for frame in gt_frames:
        g = [i for i, track in enumerate(gt_boxes) if frame in track]
        p = [i for i, track in enumerate(pred_boxes) if frame in track]
        gt_ids.append(np.asarray(g, dtype=int))
        tracker_ids.append(np.asarray(p, dtype=int))
        if g and p:
            sims.append(iou_xyxy(np.stack([gt_boxes[i][frame] for i in g]),
                                 np.stack([pred_boxes[i][frame] for i in p])))
        else:
            sims.append(np.zeros((len(g), len(p))))
    return {
        "num_gt_ids": len(gt_boxes),
        "num_tracker_ids": len(pred_boxes),
        "num_gt_dets": sum(len(t) for t in gt_boxes),
        "num_tracker_dets": sum(len(t) for t in pred_boxes),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": sims,
    }


def score_question(gt_video: dict[str, Any], question: dict[str, Any],
                   pred_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    data = question_data(gt_video["object_tracking"], [int(a) for a in question["answers"]], pred_tracks)
    res = eval_sequence(data)
    out = {k: float(np.mean(res[k])) for k in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr")}
    out.update({k: data[k] for k in ("num_gt_ids", "num_tracker_ids", "num_gt_dets", "num_tracker_dets")})
    return out


def evaluate(predictions: dict[str, Any], annotations: dict[str, Any],
             subset: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for video_id, question_id in subset:
        video = annotations[video_id]
        question = next(q for q in video["grounded_question"] if str(q["id"]) == str(question_id))
        pred = predictions.get(video_id, {}).get("grounded_question", {}).get(str(question_id))
        if pred is None:
            pred = predictions.get(video_id, {}).get("grounded_question", {}).get(int(question_id), [])
        row = {"sample": f"{video_id}_q{question_id}", "video_id": video_id, "question_id": str(question_id)}
        row.update(score_question(video, question, pred or []))
        rows.append(row)
    return rows


def load_subset(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            v, q = line.split("\t")
            rows.append((v.strip(), q.strip()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    rows = evaluate(predictions, annotations, load_subset(args.subset))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("HOTA", "DetA", "AssA", "LocA")}
    summary["questions"] = len(rows)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
