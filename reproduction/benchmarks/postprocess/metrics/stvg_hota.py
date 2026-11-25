#!/usr/bin/env python3
"""HOTA for box tracks, a dependency-light port of TrackEval's HOTA metric.

``eval_sequence`` and ``combine_sequences`` follow TrackEval
(``trackeval/metrics/hota.py``, Luiten et al., IJCV 2021) line for line: the
same 19 alpha thresholds, global alignment score, Hungarian matching on
``alignment * similarity``, and HOTA_TP-weighted combination across sequences.
tests/test_vidstg_eval.py cross-checks against TrackEval when it is installed.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo("float").eps
ARRAY_FIELDS = (
    "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA",
)
INTEGER_ARRAY_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")


def box_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """IoU of xyxy boxes, shape (len(gt), len(pred)); TrackEval's x0y0x1y1 rule."""
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    pred = np.asarray(pred, dtype=float).reshape(-1, 4)
    x0 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y0 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x1 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y1 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    intersection = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = area_gt[:, None] + area_pred[None, :] - intersection
    iou = np.zeros_like(intersection)
    mask = union > EPS
    iou[mask] = intersection[mask] / union[mask]
    return np.clip(iou, 0.0, 1.0)


def _empty_result() -> dict[str, Any]:
    res: dict[str, Any] = {
        field: np.zeros(len(ALPHAS), dtype=float)
        for field in INTEGER_ARRAY_FIELDS + ARRAY_FIELDS
    }
    return res


def _compute_final_fields(res: dict[str, Any]) -> dict[str, Any]:
    res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
    res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
    res["DetA"] = res["HOTA_TP"] / np.maximum(
        1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"]
    )
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    res["HOTA(0)"] = res["HOTA"][0]
    res["LocA(0)"] = res["LocA"][0]
    res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]
    return res


def eval_sequence(
    gt_ids: Sequence[Iterable[int]],
    tracker_ids: Sequence[Iterable[int]],
    similarity_scores: Sequence[np.ndarray],
) -> dict[str, Any]:
    """HOTA for one sequence.

    Per timestep t: ``gt_ids[t]`` / ``tracker_ids[t]`` are contiguous integer ids
    present at t, and ``similarity_scores[t]`` has shape (len(gt), len(tracker)).
    """
    gt_ids = [np.asarray(list(ids), dtype=int) for ids in gt_ids]
    tracker_ids = [np.asarray(list(ids), dtype=int) for ids in tracker_ids]
    similarity_scores = [
        np.asarray(sim, dtype=float).reshape(len(g), len(p))
        for sim, g, p in zip(similarity_scores, gt_ids, tracker_ids)
    ]
    num_gt_dets = sum(len(ids) for ids in gt_ids)
    num_tracker_dets = sum(len(ids) for ids in tracker_ids)
    res = _empty_result()
    if num_tracker_dets == 0:
        res["HOTA_FN"] = num_gt_dets * np.ones(len(ALPHAS), dtype=float)
        res["LocA"] = np.ones(len(ALPHAS), dtype=float)
        return _compute_final_fields(res)
    if num_gt_dets == 0:
        res["HOTA_FP"] = num_tracker_dets * np.ones(len(ALPHAS), dtype=float)
        res["LocA"] = np.ones(len(ALPHAS), dtype=float)
        return _compute_final_fields(res)

    num_gt_ids = 1 + max((int(ids.max()) for ids in gt_ids if len(ids)), default=-1)
    num_tracker_ids = 1 + max((int(ids.max()) for ids in tracker_ids if len(ids)), default=-1)
    potential_matches_count = np.zeros((num_gt_ids, num_tracker_ids))
    gt_id_count = np.zeros((num_gt_ids, 1))
    tracker_id_count = np.zeros((1, num_tracker_ids))
    for gt_ids_t, tracker_ids_t, similarity in zip(gt_ids, tracker_ids, similarity_scores):
        sim_iou_denom = (
            similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        )
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + EPS
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
        potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tracker_ids_t] += 1

    global_alignment_score = potential_matches_count / (
        gt_id_count + tracker_id_count - potential_matches_count
    )
    matches_counts = [np.zeros_like(potential_matches_count) for _ in ALPHAS]
    for gt_ids_t, tracker_ids_t, similarity in zip(gt_ids, tracker_ids, similarity_scores):
        if len(gt_ids_t) == 0:
            res["HOTA_FP"] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            res["HOTA_FN"] += len(gt_ids_t)
            continue
        score_mat = (
            global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]]
            * similarity
        )
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        for a, alpha in enumerate(ALPHAS):
            matched = similarity[match_rows, match_cols] >= alpha - EPS
            alpha_rows, alpha_cols = match_rows[matched], match_cols[matched]
            num_matches = len(alpha_rows)
            res["HOTA_TP"][a] += num_matches
            res["HOTA_FN"][a] += len(gt_ids_t) - num_matches
            res["HOTA_FP"][a] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                res["LocA"][a] += float(np.sum(similarity[alpha_rows, alpha_cols]))
                matches_counts[a][gt_ids_t[alpha_rows], tracker_ids_t[alpha_cols]] += 1

    for a in range(len(ALPHAS)):
        matches_count = matches_counts[a]
        ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
        res["AssA"][a] = np.sum(matches_count * ass_a) / np.maximum(1, res["HOTA_TP"][a])
        ass_re = matches_count / np.maximum(1, gt_id_count)
        res["AssRe"][a] = np.sum(matches_count * ass_re) / np.maximum(1, res["HOTA_TP"][a])
        ass_pr = matches_count / np.maximum(1, tracker_id_count)
        res["AssPr"][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res["HOTA_TP"][a])
    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _compute_final_fields(res)


def combine_sequences(all_res: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """TrackEval's dataset-level combination (sums + HOTA_TP-weighted averages)."""
    res = _empty_result()
    if not all_res:
        return _compute_final_fields(res)
    for field in INTEGER_ARRAY_FIELDS:
        res[field] = sum(seq[field] for seq in all_res.values())
    for field in ("AssRe", "AssPr", "AssA"):
        res[field] = sum(seq[field] * seq["HOTA_TP"] for seq in all_res.values()) / np.maximum(
            1.0, res["HOTA_TP"]
        )
    loca_weighted_sum = sum(seq["LocA"] * seq["HOTA_TP"] for seq in all_res.values())
    res["LocA"] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res["HOTA_TP"])
    return _compute_final_fields(res)


def summary(res: dict[str, Any]) -> dict[str, float]:
    """Alpha-averaged headline numbers as TrackEval prints them."""
    out = {field: float(np.mean(res[field])) for field in ARRAY_FIELDS}
    out["HOTA(0)"] = float(res["HOTA(0)"])
    out["LocA(0)"] = float(res["LocA(0)"])
    out["HOTALocA(0)"] = float(res["HOTALocA(0)"])
    return out
