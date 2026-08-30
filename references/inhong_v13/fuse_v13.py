#!/usr/bin/env python3
"""v13 fusion: lean grid ensemble + trusted SAM3 dense overlay (offline).

Layering (one owner per error axis — see v12_ablation_analysis.md §8):
  answer axis   adjudication.json verdicts, unchanged (v11a anchor + 3.5
                decorrelated answers; hybrid carries 3.5-answer tracks).
  variance axis lean grid fusion via fuse_tracks.fuse_sample on the 4-member
                pool {v11a, v3, inhong2_3.5, hybrid} — no visual-verify layer.
  representation axis  the SAM3 dense track (run_sam_track.py) OWNS every
                frame it covers, per target, once trusted; the grid fusion
                owns occlusion segments (SAM loses fully hidden objects,
                grid Stage2 predicts amodal boxes there) and all samples or
                targets where SAM is distrusted.

Trust gate per target (GT-free): seed alignment >= --min-seed-iou AND mean
IoU between SAM and the fused grid track on >= --min-shared shared 30-grid
frames >= --gate-iou. Distrust -> grid-only (v12-lite behaviour).

Final interpolation per target: every missing frame between consecutive
anchors is filled linearly when the gap is <= --short-gap frames, or when the
bracketing boxes overlap (IoU >= 0.3) for longer gaps — dense coverage for
the ~7%% of challenge GT frames that sit off every sampling grid.

Lifetime axis (--clip-l1-span, off by default): after interpolation each
target's track is cut to the Stage 1 `presence_spans` that the same v11a call
already produced, padded by that span's own `uncertainty_frames` plus half the
30-frame Stage 2 grid cell. Without it SAM's dense track plus interpolation
decides how long an object lives and over-extends it into frames the pipeline
never claimed the object was there, which scores as false positives on
multi-target samples. Evidence: inhong/v17/L1_SPAN_PREREG.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "inhong" / "v12"))
import fuse_tracks as ft  # noqa: E402

base = ft.base

# half a Stage 2 grid cell: the finest presence boundary anything downstream
# of the 30-frame grid can resolve, added on top of Stage 1's own uncertainty.
L1_GRID_PAD = base.DEFAULT_FRAME_STRIDE // 2

LEAN_POOL = {
    "v11a": "inhong/v11/outputs/subset100_a",
    "v3": "inhong/v3/outputs/subset100",
    "inhong2_3.5": "inhong/inhong2/outputs/subset100_v1",
    "hybrid": "inhong/v4/outputs/subset100_hybrid",
}


def load_sam(sam_root: Path, sample: str):
    d = sam_root / sample
    p = d / "sparse_predictions.json"
    m = d / "sam_meta.json"
    if not p.is_file() or not m.is_file():
        return None, None
    preds = json.loads(p.read_text())
    meta = json.loads(m.read_text())
    if meta.get("status") != "ok":
        return None, meta
    tracks: dict[int, dict[int, list]] = {}
    for f, items in preds["predictions_by_frame"].items():
        for it in items:
            tracks.setdefault(it["object_id"], {})[int(f)] = it["box_2d_yxyx_1000"]
    return tracks, meta


def grid_track_of(fused_grid: dict[int, list], oid) -> dict[int, list]:
    out = {}
    for f, dets in fused_grid.items():
        if f % base.DEFAULT_FRAME_STRIDE:
            continue
        for o, b in dets:
            if o == oid:
                out[f] = b
                break
    return out


def match_grid_oid(sam_track: dict[int, list], fused_grid: dict[int, list]):
    """Best fused-grid object for a SAM track (direct id may not exist when the
    grid base was swapped to another member's id space)."""
    grid_oids = {o for dets in fused_grid.values() for o, _ in dets}
    best, best_iou = None, 0.0
    for goid in grid_oids:
        gt = grid_track_of(fused_grid, goid)
        shared = [f for f in gt if f in sam_track]
        if len(shared) < 2:
            continue
        mean_iou = statistics.mean(ft.iou(sam_track[f], gt[f]) for f in shared)
        if mean_iou > best_iou:
            best, best_iou = goid, mean_iou
    return best, best_iou


def style_couple(sam: dict[int, list], grid: dict[int, list]) -> dict[int, list]:
    """Re-style SAM boxes toward the grid's amodal extent via anchor residuals.

    SAM tracks visible pixels; Stage2 grid boxes (and the challenge GT) are
    amodal. Raw overlay therefore bleeds a small loss on every partially
    occluded frame of ordinary samples (measured: net -0.005 vs grid on
    subset100 despite big wins on broken-grid samples). At every 30-grid
    anchor where SAM and grid agree (IoU >= 0.3) the residual grid-SAM is
    recorded; between anchors it is linearly interpolated and added to SAM's
    boxes, and ON those anchors the grid box itself is kept. Where the grid
    is wrong (different object -> no agreeing anchors) no residual exists and
    raw SAM survives untouched — the coupling degrades gracefully in exactly
    the regime SAM is there to fix. No new thresholds beyond the existing
    0.3 IoU convention."""
    anchors = sorted(f for f in grid if f in sam and ft.iou(sam[f], grid[f]) >= 0.3)
    res = {f: [grid[f][i] - sam[f][i] for i in range(4)] for f in anchors}
    out = {}
    for f, box in sam.items():
        if f in res:
            out[f] = grid[f]
            continue
        prev = max((a for a in anchors if a < f), default=None)
        nxt = min((a for a in anchors if a > f), default=None)
        if prev is not None and nxt is not None:
            w = (f - prev) / (nxt - prev)
            r = [res[prev][i] * (1 - w) + res[nxt][i] * w for i in range(4)]
        elif prev is not None and f - prev <= 60:
            r = res[prev]
        elif nxt is not None and nxt - f <= 60:
            r = res[nxt]
        else:
            r = None
        out[f] = _sane([box[i] + r[i] for i in range(4)], box) if r else box
    return out


def _sane(b: list, fallback: list) -> list:
    y0, x0, y1, x1 = (min(1000.0, max(0.0, v)) for v in b)
    if y1 - y0 >= 2 and x1 - x0 >= 2:
        return [y0, x0, y1, x1]
    return fallback


def interpolate_track(track: dict[int, list], short_gap: int) -> dict[int, list]:
    frames = sorted(track)
    out = dict(track)
    for p, n in zip(frames, frames[1:]):
        gap = n - p
        if gap <= 1:
            continue
        if gap > short_gap and ft.iou(track[p], track[n]) < 0.3:
            continue
        for f in range(p + 1, n):
            w = (f - p) / gap
            out[f] = [track[p][i] * (1 - w) + track[n][i] * w for i in range(4)]
    return out


def load_l1_spans(plan_root: Path, sample: str) -> dict[int, list[tuple[int, int]]]:
    """Stage 1 presence spans per target, in padded source-frame indices.

    Pad = the span's own uncertainty_frames (Stage 1 states its boundary error
    bar) + half the Stage 2 grid cell, which is the finest boundary anything
    downstream can resolve. No GT, no extra API call — the plan JSON is already
    written by the v11a run this fusion consumes.
    """
    plan_p = plan_root / sample / "gemini_stage1_plan.json"
    man_p = plan_root / sample / "manifest.json"
    if not (plan_p.is_file() and man_p.is_file()):
        return {}
    fps = json.loads(man_p.read_text())["video"]["fps"]
    plan = json.loads(plan_p.read_text())
    plan = plan.get("response", plan)
    out: dict[int, list[tuple[int, int]]] = {}
    for target in plan.get("targets", []):
        spans = []
        for s in target.get("presence_spans", []):
            pad = int(s.get("uncertainty_frames", 0) or 0) + L1_GRID_PAD
            spans.append((round(s["start_time_seconds"] * fps) - pad,
                          round(s["end_time_seconds"] * fps) + pad))
        if spans:
            out[int(target["object_id"])] = spans
    return out


def clip_to_spans(track: dict[int, list],
                  spans: list[tuple[int, int]]) -> dict[int, list]:
    """Keep only frames inside a presence span; never empty a track."""
    kept = {f: b for f, b in track.items()
            if any(lo <= f <= hi for lo, hi in spans)}
    return kept or track


def fuse_sample_v13(sample, sam_root, adjudication, args):
    fused_grid, gmeta = ft.fuse_sample(sample, LEAN_POOL, adjudication, {})
    sam_tracks, smeta = load_sam(Path(sam_root), sample)
    per_target = []
    final: dict[int, dict[int, list]] = {}

    sam_by_target = {}
    if sam_tracks and smeta:
        seed_iou = {t["object_id"]: t.get("seed_alignment_iou", 0.0)
                    for t in smeta.get("targets", [])}
        for oid, track in sam_tracks.items():
            goid = oid
            gt = grid_track_of(fused_grid, goid)
            shared = [f for f in gt if f in track]
            if len(shared) < args.min_shared:
                goid, _ = match_grid_oid(track, fused_grid)
                if goid is None:
                    per_target.append({"sam_oid": oid, "trusted": False,
                                       "reason": "no_grid_match"})
                    continue
                gt = grid_track_of(fused_grid, goid)
                shared = [f for f in gt if f in track]
            corr = (statistics.mean(ft.iou(track[f], gt[f]) for f in shared)
                    if shared else 0.0)
            trusted = (seed_iou.get(oid, 0.0) >= args.min_seed_iou
                       and len(shared) >= args.min_shared
                       and corr >= args.gate_iou)
            per_target.append({"sam_oid": oid, "grid_oid": goid,
                               "trusted": trusted, "corr": round(corr, 3),
                               "n_shared": len(shared),
                               "seed_iou": round(seed_iou.get(oid, 0.0), 3)})
            if trusted:
                sam_by_target[goid] = track

    l1_spans = (load_l1_spans(Path(args.l1_plan_root), sample)
                if args.clip_l1_span else {})
    n_clipped = 0

    grid_oids = {o for dets in fused_grid.values() for o, _ in dets}
    for goid in grid_oids:
        gtrack = grid_track_of(fused_grid, goid)
        merged = dict(gtrack)
        if goid in sam_by_target:
            styled = style_couple(sam_by_target[goid], gtrack)
            merged.update(styled)
        track = interpolate_track(merged, args.short_gap)
        if goid in l1_spans:
            before = len(track)
            track = clip_to_spans(track, l1_spans[goid])
            n_clipped += before - len(track)
        final[goid] = track

    fused: dict[int, list] = {}
    for goid, track in final.items():
        for f, b in track.items():
            fused.setdefault(f, []).append((goid, b))
    meta = {"grid": gmeta, "sam_targets": per_target,
            "sam_used": sorted(sam_by_target),
            "l1_clipped_frames": n_clipped,
            "l1_spans_used": sorted(l1_spans)}
    return fused, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default=str(
        PROJECT_ROOT / "minseon" / "pipeline-v3" / "subset100.tsv"))
    parser.add_argument("--sam-root", required=True)
    parser.add_argument("--adjudication", default=str(
        PROJECT_ROOT / "inhong" / "v12" / "adjudication.json"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pool", default=None,
                        help="JSON dict overriding the lean grid pool")
    parser.add_argument("--gate-iou", type=float, default=0.30)
    parser.add_argument("--min-seed-iou", type=float, default=0.25)
    parser.add_argument("--min-shared", type=int, default=3)
    parser.add_argument("--short-gap", type=int, default=6)
    parser.add_argument("--clip-l1-span", action="store_true",
                        help="cut each track to its Stage 1 presence span")
    parser.add_argument("--l1-plan-root", default=None,
                        help="root holding gemini_stage1_plan.json + "
                             "manifest.json (defaults to the pool's v11a run)")
    args = parser.parse_args()

    global LEAN_POOL
    if args.pool:
        LEAN_POOL = json.loads(args.pool)
    if args.l1_plan_root is None:
        args.l1_plan_root = LEAN_POOL["v11a"]
    if not Path(args.l1_plan_root).is_absolute():
        args.l1_plan_root = str(PROJECT_ROOT / args.l1_plan_root)
    if args.clip_l1_span and not Path(args.l1_plan_root).is_dir():
        parser.error(f"--l1-plan-root not found: {args.l1_plan_root}")
    adjudication = {}
    if Path(args.adjudication).is_file():
        adjudication = json.loads(Path(args.adjudication).read_text())

    subset = [tuple(p.strip() for p in line.split("\t"))
              for line in Path(args.subset).read_text().strip().splitlines()]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for video_id, qid in subset:
        sample = f"{video_id}_q{qid}"
        fused, meta = fuse_sample_v13(sample, args.sam_root, adjudication, args)
        strict_report, challenge = ft.score(fused, video_id, qid)
        d = out_root / sample
        d.mkdir(parents=True, exist_ok=True)
        base.save_json(d / "sparse_predictions.json", {
            "fusion": {"v13": meta["sam_used"], "targets": meta["sam_targets"],
                       "grid_base": meta["grid"]["base"]},
            "frame_stride": base.DEFAULT_FRAME_STRIDE,
            "predictions_by_frame": {
                str(f): [{"object_id": oid, "visibility": "visible",
                          "confidence": 1.0,
                          "box_2d_yxyx_1000": list(map(round, b)),
                          "xyxy_normalized": base.yxyx_1000_to_xyxy(b)}
                         for oid, b in dets]
                for f, dets in sorted(fused.items())},
        })
        base.save_json(d / "hota_sparse_gemini.json", strict_report)
        rows.append({"sample": sample, "challenge": challenge,
                     "sam_used": meta["sam_used"],
                     "targets": meta["sam_targets"],
                     "grid_base": meta["grid"]["base"]})
        print(f"{sample} sam={len(meta['sam_used'])} "
              f"challenge={challenge if challenge is None else round(challenge, 3)}",
              flush=True)
    ch = [r["challenge"] for r in rows if r["challenge"] is not None]
    n_sam = sum(1 for r in rows if r["sam_used"])
    summary = {"num_samples": len(rows), "num_sam_overlaid": n_sam,
               "challenge_mean": statistics.mean(ch) if ch else None}
    base.save_json(out_root / "fusion_summary.json",
                   {"summary": summary, "rows": rows})
    print(f"\nchallenge mean = {summary['challenge_mean']:.4f} "
          f"(SAM overlaid on {n_sam}/{len(rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
