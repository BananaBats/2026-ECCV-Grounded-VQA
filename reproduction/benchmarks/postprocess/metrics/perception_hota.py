#!/usr/bin/env python3
"""HOTA for one question, following TrackEval ``trackeval/metrics/hota.py``.

Same 19 alpha thresholds (0.05..0.95), global alignment score, Hungarian
matching on ``alignment * similarity``, and per-alpha TP/FN/FP accumulation.
Input layout is TrackEval's ``data`` dict as built by the official Perception
Test grounded-VQA notebook (``run_iou``): ``gt_ids[t]``, ``tracker_ids[t]``,
``similarity_scores[t]`` (|gt_t| x |pred_t| IoU), ``num_gt_ids``,
``num_tracker_ids``, ``num_gt_dets``, ``num_tracker_dets``.
tests/test_val200_eval.py cross-checks against the TrackEval source checkout
when it is importable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo("float").eps
FLOAT_FIELDS = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA")
INT_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")


def _final_fields(res: dict[str, Any]) -> dict[str, Any]:
    res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
    res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
    res["DetA"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"])
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    res["HOTA(0)"] = res["HOTA"][0]
    res["LocA(0)"] = res["LocA"][0]
    res["HOTALocA(0)"] = res["HOTA"][0] * res["LocA"][0]
    return res


def eval_sequence(data: dict[str, Any]) -> dict[str, Any]:
    res: dict[str, Any] = {f: np.zeros(len(ALPHAS), dtype=float) for f in INT_FIELDS + FLOAT_FIELDS}
    if data["num_tracker_dets"] == 0:
        res["HOTA_FN"] = data["num_gt_dets"] * np.ones(len(ALPHAS), dtype=float)
        res["LocA"] = np.ones(len(ALPHAS), dtype=float)
        res["LocA(0)"] = 1.0
        return _final_fields(res)
    if data["num_gt_dets"] == 0:
        res["HOTA_FP"] = data["num_tracker_dets"] * np.ones(len(ALPHAS), dtype=float)
        res["LocA"] = np.ones(len(ALPHAS), dtype=float)
        res["LocA(0)"] = 1.0
        return _final_fields(res)

    num_gt, num_tr = int(data["num_gt_ids"]), int(data["num_tracker_ids"])
    potential = np.zeros((num_gt, num_tr))
    gt_count = np.zeros((num_gt, 1))
    tr_count = np.zeros((1, num_tr))
    for gt_t, tr_t, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        gt_t, tr_t = np.asarray(gt_t, dtype=int), np.asarray(tr_t, dtype=int)
        sim = np.asarray(sim, dtype=float).reshape(len(gt_t), len(tr_t))
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        sim_iou = np.zeros_like(sim)
        mask = denom > 0 + EPS
        sim_iou[mask] = sim[mask] / denom[mask]
        potential[gt_t[:, None], tr_t[None, :]] += sim_iou
        gt_count[gt_t] += 1
        tr_count[0, tr_t] += 1
    global_alignment = potential / (gt_count + tr_count - potential)
    matches = [np.zeros_like(potential) for _ in ALPHAS]

    for gt_t, tr_t, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        gt_t, tr_t = np.asarray(gt_t, dtype=int), np.asarray(tr_t, dtype=int)
        if len(gt_t) == 0:
            res["HOTA_FP"] += len(tr_t)
            continue
        if len(tr_t) == 0:
            res["HOTA_FN"] += len(gt_t)
            continue
        sim = np.asarray(sim, dtype=float).reshape(len(gt_t), len(tr_t))
        score = global_alignment[gt_t[:, None], tr_t[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            keep = sim[rows, cols] >= alpha - EPS
            r, c = rows[keep], cols[keep]
            n = len(r)
            res["HOTA_TP"][a] += n
            res["HOTA_FN"][a] += len(gt_t) - n
            res["HOTA_FP"][a] += len(tr_t) - n
            if n:
                res["LocA"][a] += float(sim[r, c].sum())
                matches[a][gt_t[r], tr_t[c]] += 1

    for a in range(len(ALPHAS)):
        m = matches[a]
        ass_a = m / np.maximum(1, gt_count + tr_count - m)
        res["AssA"][a] = (m * ass_a).sum() / max(1, res["HOTA_TP"][a])
        ass_re = m / np.maximum(1, gt_count)
        res["AssRe"][a] = (m * ass_re).sum() / max(1, res["HOTA_TP"][a])
        ass_pr = m / np.maximum(1, tr_count)
        res["AssPr"][a] = (m * ass_pr).sum() / max(1, res["HOTA_TP"][a])
    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _final_fields(res)
