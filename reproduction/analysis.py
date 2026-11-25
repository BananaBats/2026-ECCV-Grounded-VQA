"""The shared-threshold and withheld-frame experiments on Perception Test."""
from bisect import bisect_left, bisect_right

import numpy as np

from .benchmarks.postprocess import core
from .benchmarks.postprocess.controls import counterfactual_correction
from .benchmarks.postprocess.metrics.perception_eval import gt_track_start, score_question


METRICS = ("HOTA", "DetA", "AssA", "LocA")
SHARED_TAUS = [i / 1000 for i in range(1001)]
HOLDOUT_TAUS = [i / 20 for i in range(21)]


def scored_gt(tracks, frames=None):
    result = []
    for track in tracks:
        start = gt_track_start(track)
        obs = [(f, box) for f, box in zip(track["frame_ids"][start:], track["bounding_boxes"][start:])
               if frames is None or f in frames]
        result.append({"id": track["id"], "frame_ids": [f for f, _ in obs],
                       "bounding_boxes": [b for _, b in obs]})
    return {"object_tracking": result}


def score(gt, answers, tracks):
    pred = core.official_payload("clip", "0", tracks)["clip"]["grounded_question"]["0"]
    result = score_question(gt, {"answers": answers}, pred)
    return [result[k] * 100 for k in METRICS]


def shared_curve(sam, gemini, gt, answers):
    """The output on annotated frames changes only when an IoU crosses a gate."""
    taus = SHARED_TAUS
    gt_frames = sorted({f for track in gt["object_tracking"] for f in track["frame_ids"]})
    residual_cuts = {0, len(taus)}
    for oid, sparse in gemini.items():
        dense = sam.get(oid, {})
        for f in dense.keys() & sparse.keys():
            residual_cuts.add(bisect_right(taus, core.box_iou(dense[f], sparse[f])))

    scores = np.empty(len(taus))
    cache = {}
    cuts = sorted(residual_cuts)
    for first, stop in zip(cuts, cuts[1:]):
        candidates = {}
        gap_cuts = {first, stop}
        for oid, sparse in sorted(gemini.items()):
            corrected, _ = core.correct_sam_track(sam.get(oid, {}), sparse, taus[first])
            merged = {**sparse, **corrected}
            if not merged:
                continue
            frames = sorted(merged)
            candidates[oid] = []
            for f in gt_frames:
                if f in merged:
                    candidates[oid].append((f, merged[f], stop))
                    continue
                i = bisect_left(frames, f)
                if i == 0 or i == len(frames):
                    continue
                a, b = frames[i - 1], frames[i]
                end = min(stop, bisect_right(taus, core.box_iou(merged[a], merged[b])))
                if end <= first:
                    continue
                w = (f - a) / (b - a)
                box = [x * (1 - w) + y * w for x, y in zip(merged[a], merged[b])]
                candidates[oid].append((f, box, end))
                gap_cuts.add(end)
        boundaries = sorted(gap_cuts)
        for left, right in zip(boundaries, boundaries[1:]):
            pred = {oid: {f: box for f, box, end in track if left < end}
                    for oid, track in candidates.items()}
            signature = tuple((oid, tuple((f, tuple(b)) for f, b in track.items()))
                              for oid, track in pred.items())
            if signature not in cache:
                cache[signature] = score(gt, answers, pred)[0]
            scores[left:right] = cache[signature]
    return scores


def holdout(sam, gemini, queried, gt, answers):
    """Hide entire queried timestamps by parity, including empty Gemini replies."""
    gt_frames = {f for track in gt["object_tracking"] for f in track["frame_ids"]}
    evaluation = set()
    predictions = [{oid: {} for oid in sam} for _ in HOLDOUT_TAUS]
    baseline = {oid: {} for oid in sam}
    no_expansion = {oid: {} for oid in sam}
    for parity in (0, 1):
        hidden = set(queried[parity::2])
        retained = {oid: {f: b for f, b in track.items() if f not in hidden}
                    for oid, track in gemini.items()}
        # Eligibility is fixed using ungated, same-ID references on both sides.
        frames = set()
        for oid, dense in sam.items():
            shared = sorted(dense.keys() & retained.get(oid, {}).keys())
            if len(shared) < 2:
                continue
            frames.update(f for f in dense.keys() & hidden & gt_frames if shared[0] < f < shared[-1])
        evaluation.update(frames)
        for oid, dense in sam.items():
            support = dense.keys() & frames
            baseline[oid].update({f: dense[f] for f in support})
            training = retained.get(oid, {})
            for i, tau in enumerate(HOLDOUT_TAUS):
                corrected, refs = core.correct_sam_track(dense, training, tau)
                predictions[i][oid].update({f: corrected[f] for f in support})
                if tau == 0.3:
                    smaller = counterfactual_correction(dense, training, refs)
                    no_expansion[oid].update({f: smaller[f] for f in support})
    if not evaluation:
        return None
    support = {oid: set(track) for oid, track in baseline.items()}
    for arm in [no_expansion, *predictions]:
        if {oid: set(track) for oid, track in arm.items()} != support:
            raise ValueError("Holdout arms have different output support")
    restricted = scored_gt(gt["object_tracking"], evaluation)
    curves = [score(restricted, answers, tracks) for tracks in predictions]
    return len(evaluation), curves, score(restricted, answers, baseline), score(restricted, answers, no_expansion)


def paired_ci(differences, videos):
    """Question-macro paired mean, resampling whole videos with replacement."""
    clusters = sorted(set(videos))
    totals = np.array([sum(value for value, v in zip(differences, videos) if v == c) for c in clusters])
    counts = np.array([videos.count(c) for c in clusters])
    draw = np.random.default_rng(20261119).integers(len(clusters), size=(5000, len(clusters)))
    means = totals[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def shared_statistics(scores, videos):
    """Pointwise paired bootstrap intervals relative to the reported threshold."""
    scores = np.asarray(scores)
    baseline = scores[:, SHARED_TAUS.index(0.3)]
    diffs = scores - baseline[:, None]
    clusters = sorted(set(videos))
    groups = np.array([clusters.index(video) for video in videos])
    sums = np.stack([diffs[groups == g].sum(axis=0) for g in range(len(clusters))])
    counts = np.bincount(groups)
    rng = np.random.default_rng(20261119)
    prob = np.full(len(clusters), 1 / len(clusters))
    estimates = []
    for _ in range(20):
        weights = rng.multinomial(len(clusters), prob, size=250)
        estimates.append((weights @ sums) / (weights @ counts)[:, None])
    lower, upper = np.quantile(np.concatenate(estimates), [0.025, 0.975], axis=0)
    mean = scores.mean(axis=0)
    return np.column_stack((mean, diffs.mean(axis=0), mean.max() - mean, lower, upper))
