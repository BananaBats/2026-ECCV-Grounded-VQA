#!/usr/bin/env python3
"""VidSTG evaluation: HOTA (TrackEval-style) plus the paper's tIoU / vIoU.

Each sentence is one tracking sequence over its evaluated segment
[used_segment.begin_fid, used_segment.end_fid).  GT has one track (the target
tube, temporal_gt clipped to the segment); the prediction is the exported tube.
A predicted box outside the GT tube is a false positive and a missing box
inside it is a false negative, so HOTA scores space and time together.

Reported per split (all / declarative / interrogative):
  HOTA, DetA, AssA, LocA, DetRe, DetPr   TrackEval combine_sequences over sentences
  HOTA_mean_per_sentence                 unweighted mean of per-sentence HOTA
  m_tIoU, m_vIoU, vIoU@0.3, vIoU@0.5     VidSTG paper 5.1 (vIoU@R counts vIoU > R)

--oracle-temporal drops predicted boxes outside the GT tube and uses the GT
segment as the predicted segment (the paper's "+ Tem. GT" setting).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from . import stvg_hota as hota  # noqa: E402


SPLITS = ("all", "declarative", "interrogative")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def single_iou(a: list[float], b: list[float]) -> float:
    return float(hota.box_iou_matrix(np.asarray([a]), np.asarray([b]))[0, 0])


def sequence_data(
    gt_boxes: dict[str, list[float]],
    pred_boxes: dict[str, list[float]],
    frames: range,
) -> tuple[list[list[int]], list[list[int]], list[np.ndarray]]:
    gt_ids: list[list[int]] = []
    tracker_ids: list[list[int]] = []
    similarity: list[np.ndarray] = []
    for frame in frames:
        gt = gt_boxes.get(str(frame))
        pred = pred_boxes.get(str(frame))
        gt_ids.append([0] if gt is not None else [])
        tracker_ids.append([0] if pred is not None else [])
        if gt is not None and pred is not None:
            similarity.append(np.asarray([[single_iou(gt, pred)]]))
        else:
            similarity.append(np.zeros((1 if gt is not None else 0, 1 if pred is not None else 0)))
    return gt_ids, tracker_ids, similarity


def temporal_iou(a: tuple[int, int] | None, b: tuple[int, int]) -> float:
    if a is None:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def viou(
    gt_boxes: dict[str, list[float]],
    pred_boxes: dict[str, list[float]],
    gt_segment: tuple[int, int],
    pred_segment: tuple[int, int] | None,
    stride: int,
) -> float:
    """(1/|S_U|) * sum_{t in S_I} IoU(pred_t, gt_t) over the stride grid."""
    if pred_segment is None:
        return 0.0
    gt_frames = set(range(gt_segment[0], gt_segment[1]))
    pred_frames = set(range(pred_segment[0], pred_segment[1]))
    grid = lambda frames: {frame for frame in frames if frame % stride == 0}  # noqa: E731
    union = grid(gt_frames | pred_frames)
    if not union:
        return 0.0
    total = 0.0
    for frame in grid(gt_frames & pred_frames):
        gt = gt_boxes.get(str(frame))
        pred = pred_boxes.get(str(frame))
        if gt is not None and pred is not None:
            total += single_iou(gt, pred)
    return total / len(union)


def evaluate_sample(
    gt: dict[str, Any],
    prediction: dict[str, Any] | None,
    *,
    stride: int,
    oracle_temporal: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    eval_begin, eval_end = (int(value) for value in gt["eval_frames"])
    tube = (int(gt["tube"][0]), int(gt["tube"][1]))
    gt_boxes = {
        frame: box for frame, box in gt["boxes_xyxy"].items()
        if tube[0] <= int(frame) < tube[1]
    }
    pred_boxes = dict((prediction or {}).get("boxes_xyxy") or {})
    raw_segment = (prediction or {}).get("pred_segment")
    pred_segment = tuple(int(value) for value in raw_segment) if raw_segment else None
    if oracle_temporal:
        pred_boxes = {
            frame: box for frame, box in pred_boxes.items() if tube[0] <= int(frame) < tube[1]
        }
        pred_segment = tube
    pred_boxes = {
        frame: box for frame, box in pred_boxes.items() if eval_begin <= int(frame) < eval_end
    }
    frames = range(eval_begin + (-eval_begin) % stride, eval_end, stride)
    hota_res = hota.eval_sequence(*sequence_data(gt_boxes, pred_boxes, frames))
    scores = {
        "tIoU": temporal_iou(pred_segment, tube),
        "vIoU": viou(gt_boxes, pred_boxes, tube, pred_segment, stride),
        "HOTA": float(np.mean(hota_res["HOTA"])),
    }
    return hota_res, scores


def aggregate(
    results: dict[str, tuple[dict[str, Any], dict[str, float]]]
) -> dict[str, Any]:
    if not results:
        return {"num_sentences": 0}
    combined = hota.summary(hota.combine_sequences({k: v[0] for k, v in results.items()}))
    vious = np.asarray([scores["vIoU"] for _, scores in results.values()])
    return {
        "num_sentences": len(results),
        **{key: combined[key] for key in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr")},
        "HOTA_mean_per_sentence": float(np.mean([scores["HOTA"] for _, scores in results.values()])),
        "m_tIoU": float(np.mean([scores["tIoU"] for _, scores in results.values()])),
        "m_vIoU": float(vious.mean()),
        "vIoU@0.3": float((vious > 0.3).mean()),
        "vIoU@0.5": float((vious > 0.5).mean()),
    }


def evaluate(
    gt: dict[str, Any],
    predictions: dict[str, Any],
    *,
    stride: int = 1,
    oracle_temporal: bool = False,
    samples: set[str] | None = None,
) -> dict[str, Any]:
    per_split: dict[str, dict[str, tuple[dict[str, Any], dict[str, float]]]] = {
        split: {} for split in SPLITS
    }
    per_sample: dict[str, dict[str, float]] = {}
    missing = []
    for sample, record in sorted(gt.items()):
        if samples is not None and sample not in samples:
            continue
        prediction = predictions.get(sample)
        if prediction is None:
            missing.append(sample)
        result = evaluate_sample(record, prediction, stride=stride, oracle_temporal=oracle_temporal)
        per_split["all"][sample] = result
        per_split[record["sentence_type"]][sample] = result
        per_sample[sample] = {**result[1], "sentence_type": record["sentence_type"]}
    return {
        "settings": {"frame_stride": stride, "oracle_temporal": oracle_temporal},
        "num_missing_predictions": len(missing),
        "missing_predictions": missing[:50],
        "metrics": {split: aggregate(per_split[split]) for split in SPLITS},
        "per_sample": per_sample,
    }


def format_table(report: dict[str, Any]) -> str:
    keys = ("HOTA", "DetA", "AssA", "LocA", "m_tIoU", "m_vIoU", "vIoU@0.3", "vIoU@0.5")
    lines = ["split          n      " + "  ".join(f"{key:>8}" for key in keys)]
    for split in SPLITS:
        metrics = report["metrics"][split]
        if not metrics.get("num_sentences"):
            continue
        values = "  ".join(f"{100 * metrics[key]:8.2f}" for key in keys)
        lines.append(f"{split:<13}{metrics['num_sentences']:>6}  {values}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, required=True, help="dataset/<split>/eval/vidstg_gt.json")
    parser.add_argument("--predictions", type=Path, required=True, help="vidstg_export.py output")
    parser.add_argument("--subset", type=Path,
                        help="only score these clip_id<TAB>question_id rows (e.g. a pilot)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="evaluate every Nth original frame (1 = all annotated frames)")
    parser.add_argument("--oracle-temporal", action="store_true")
    parser.add_argument("--output", type=Path, help="write the full JSON report here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gt = load_json(args.gt)
    payload = load_json(args.predictions)
    predictions = payload.get("samples", payload)
    samples = None
    if args.subset is not None:
        samples = set()
        for line in args.subset.read_text(encoding="utf-8").splitlines():
            if line.strip():
                clip_id, question_id = line.split("\t")
                samples.add(f"{clip_id.strip()}_q{question_id.strip()}")
    report = evaluate(
        gt, predictions, stride=args.frame_stride,
        oracle_temporal=args.oracle_temporal, samples=samples,
    )
    report["settings"]["temporal_source"] = payload.get("temporal_source")
    print(format_table(report))
    print(f"missing predictions: {report['num_missing_predictions']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
