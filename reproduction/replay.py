"""Recompute the val200 extent ablation, tied IoU sweep and temporal holdout."""
import argparse
import csv
import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

from sdtrack.fusion import fuse
from .analysis import (HOLDOUT_TAUS, METRICS, SHARED_TAUS, holdout, paired_ci, score,
                       scored_gt, shared_curve, shared_statistics)
from .benchmarks.postprocess.fusion import fuse_tracks
from .benchmarks.postprocess.metrics.perception_eval import score_question
from .select_val200 import DATA, select


REFERENCE = Path(__file__).resolve().parent / "reference"
SHARED_COLUMNS = ("HOTA_points", "delta_vs03_points", "regret_to_observed_best_points",
                  "pointwise_CI95_lower_delta", "pointwise_CI95_upper_delta")


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def unpack(raw):
    return {int(oid): {int(f): box for f, box in tr.items()} for oid, tr in raw.items()}


def api_cost():
    """Standard-rate estimates used in the paper, including received retries."""
    stages = dict.fromkeys(("plan", "sparse", "direct"), Decimal(0))
    seen = set()
    with gzip.open(DATA / "usage_counts.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["attempt_id"] in seen:
                raise ValueError("Duplicate model response in the token ledger")
            seen.add(row["attempt_id"])
            u = row.get("usage_metadata") or {}
            prompt, cached = (u.get(k) or 0 for k in ("prompt_token_count", "cached_content_token_count"))
            out = (u.get("candidates_token_count") or 0) + (u.get("thoughts_token_count") or 0)
            stages[row["stage"]] += ((prompt - cached) * Decimal("0.75") + cached * Decimal("0.075")
                                     + out * Decimal("3.75")) / 1_000_000
    shared = float(stages["plan"] + stages["sparse"])
    return {"dense": float(stages["plan"]), "sparse_linear": shared, "full": shared,
            "direct": float(stages["direct"])}


def checked_curve(file, threshold_key, thresholds, score_keys, values):
    with (REFERENCE / file).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row, tau in zip(rows, thresholds):
        if float(row[threshold_key]) != tau:
            raise ValueError(f"Different threshold grid in {file}")
    expected = [[float(row[k]) for k in score_keys] for row in rows]
    np.testing.assert_allclose(values, expected, atol=1e-8, rtol=0)


def replay():
    selected = select()
    if selected.encode() != (DATA / "perception_val200.tsv").read_bytes():
        raise ValueError("Selection differs from the archived query list")
    cohort = read_gzip(DATA / "val200.json.gz")
    maps = read_gzip(DATA / "val200_frame_maps.json.gz")
    direct = read_gzip(DATA / "direct_on_gt.json.gz")
    ordered = [(q["video_id"], q["question_id"]) for q in cohort]
    if ordered != [tuple(line.split("\t")) for line in selected.splitlines()]:
        raise ValueError("Cached inputs differ from the selected cohort")
    native, shared, withheld, baselines, no_extent = [], [], [], [], []
    eligible_videos = []
    nboxes = nslots = annotated = on_grid = 0
    for q in cohort:
        video, qid = q["video_id"], q["question_id"]
        gt = scored_gt(q["gt"])
        frames = {f for track in gt["object_tracking"] for f in track["frame_ids"]}
        queried = maps[f"{video}_q{qid}"]["gemini"]
        if queried != sorted(set(queried)):
            raise ValueError(f"Invalid sparse query clock for {video} question {qid}")
        annotated += len(frames)
        on_grid += len(frames.intersection(queried))
        sam, gemini = unpack(q["sam3"]), unpack(q["gemini"])
        plan = {"targets": [{"obj_id": i} for i in q["target_ids"]]}
        predictions = fuse(q, plan, sam, gemini)[video]["grounded_question"][qid]
        encoded = json.dumps(predictions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != q["full_sha256"]:
            raise ValueError(f"Full output differs from archive for {video} question {qid}")
        nboxes += sum(len(t["frame_ids"]) for t in predictions)
        ablated, _ = fuse_tracks(sam, gemini, mode="no_expansion_fixed_support")
        if {i: sorted(t) for i, t in ablated.items()} != {t["id"]: t["frame_ids"] for t in predictions}:
            raise ValueError("Extent control changed the native frame support")
        sparse_only, _ = fuse_tracks(sam, gemini, mode="sparse_only")
        native.append([score_question(gt, q, q["dense_baseline_on_gt"])["HOTA"] * 100,
                       score(gt, q["answers"], sparse_only)[0],
                       score_question(gt, q, predictions)["HOTA"] * 100,
                       score(gt, q["answers"], ablated)[0],
                       score_question(gt, q, direct[f"{video}_q{qid}"])["HOTA"] * 100])
        shared.append(shared_curve(sam, gemini, gt, q["answers"]))
        diagnostic = holdout(sam, gemini, queried, gt, q["answers"])
        if diagnostic is not None:
            count, curve, raw, smaller = diagnostic
            eligible_videos.append(video)
            nslots += count
            withheld.append(curve)
            baselines.append(raw)
            no_extent.append(smaller)

    means = np.mean(native, axis=0)
    shared_results = shared_statistics(shared, [v for v, _ in ordered])
    tied = shared_results[:, 0]
    hold = np.mean(withheld, axis=0)
    raw, smaller = np.mean(baselines, axis=0)[0], np.mean(no_extent, axis=0)[0]
    checked_curve("shared_tau_full_curve.csv", "shared_tau", SHARED_TAUS, SHARED_COLUMNS, shared_results)
    checked_curve("residual_holdout_full_curve.csv", "residual_tau", HOLDOUT_TAUS, METRICS, hold)
    np.testing.assert_allclose(means, [66.8610073712, 65.6110407856, 71.8017957068, 69.1695732127,
                                      72.5495992687], atol=1e-8, rtol=0)
    np.testing.assert_allclose([raw, smaller], [78.69031482591404, 77.36966238872306], atol=1e-8, rtol=0)
    peak = int(np.argmax(tied))
    plateaus = {}
    for margin in [0.1, 0.25]:
        left = right = SHARED_TAUS.index(0.3)
        while left > 0 and tied[left - 1] >= tied[peak] - margin:
            left -= 1
        while right < len(tied) - 1 and tied[right + 1] >= tied[peak] - margin:
            right += 1
        plateaus[str(margin)] = [SHARED_TAUS[left], SHARED_TAUS[right]]
    at_reported = np.asarray(withheld)[:, HOLDOUT_TAUS.index(0.3), 0]
    counts = [len(cohort), len(set(v for v, _ in ordered)), nboxes, annotated, on_grid,
              len(withheld), len(set(eligible_videos)), nslots]
    if counts != [200, 190, 183035, 4139, 3854, 184, 178, 2958]:
        raise ValueError(f"Cohort or support differs from the archive: {counts}")
    summary = {
        "questions": len(cohort), "videos": counts[1], "full_boxes_matched": nboxes,
        "annotated_frames": annotated, "annotated_frames_on_gemini_grid": on_grid,
        "native_HOTA": dict(zip(("dense", "sparse_linear", "full", "no_positive_expansion", "direct"), means.tolist())),
        "estimated_API_cost_USD": api_cost(),
        "shared_threshold": {"points_checked": len(tied), "best_tau": SHARED_TAUS[peak],
                             "best_HOTA": float(tied[peak]), "connected_intervals": plateaus,
                             "HOTA": {str(t): float(tied[SHARED_TAUS.index(t)]) for t in (0, 0.201, 0.3, 1)}},
        "holdout": {"questions": len(withheld), "videos": len(set(eligible_videos)), "query_frames": nslots,
                    "points_checked": len(hold), "raw_SAM_HOTA": float(raw),
                    "residual_HOTA": {str(t): float(hold[HOLDOUT_TAUS.index(t), 0]) for t in (0, 0.3, 0.8, 1)},
                    "no_positive_expansion_HOTA": float(smaller),
                    "residual_minus_raw_CI95": paired_ci(at_reported - np.asarray(baselines)[:, 0], eligible_videos),
                    "residual_minus_no_expansion_CI95": paired_ci(at_reported - np.asarray(no_extent)[:, 0], eligible_videos)},
    }
    for arm, cost in {"dense": 7.54044225, "sparse_linear": 20.90977575,
                      "full": 20.90977575, "direct": 206.18408925}.items():
        np.testing.assert_allclose(summary["estimated_API_cost_USD"][arm], cost, atol=1e-9, rtol=0)
    np.testing.assert_allclose(summary["holdout"]["residual_minus_raw_CI95"],
                               [0.2312800822, 1.1658997904], atol=1e-8, rtol=0)
    np.testing.assert_allclose(summary["holdout"]["residual_minus_no_expansion_CI95"],
                               [1.7286475211, 2.3578128715], atol=1e-8, rtol=0)
    return summary, shared_results, hold


def write_curve(path, columns, thresholds, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows((tau, *row) for tau, row in zip(thresholds, values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="save summary.json and both recomputed curves")
    args = parser.parse_args()
    results, shared, hold = replay()
    summary = json.dumps(results, indent=2, allow_nan=False)
    print(summary)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(summary + "\n", encoding="utf-8")
        write_curve(args.output_dir / "shared_tau.csv", ("shared_tau", *SHARED_COLUMNS), SHARED_TAUS, shared)
        write_curve(args.output_dir / "holdout.csv", ("residual_tau", *METRICS), HOLDOUT_TAUS, hold)
