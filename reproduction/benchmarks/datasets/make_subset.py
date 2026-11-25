#!/usr/bin/env python3
"""Stratified seed-0 subset of Perception Test validation grounded questions.

Strata = (target count bucket: 1 / 2+) x (question area). Allocation is
proportional with largest-remainder rounding; sampling inside a stratum uses
random.Random(seed). Writes config/val200.tsv and config/val200_meta.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=RELEASE_ROOT / "dataset/valid/annotations/grounded_question_valid.json")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsv", type=Path, default=RELEASE_ROOT / "config/val200.tsv")
    parser.add_argument("--meta", type=Path, default=RELEASE_ROOT / "config/val200_meta.json")
    args = parser.parse_args()

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for video_id in sorted(data):
        video = data[video_id]
        meta = video["metadata"]
        tracks = {int(t["id"]): t for t in video["object_tracking"]}
        for q in sorted(video["grounded_question"], key=lambda q: int(q["id"])):
            answers = [int(a) for a in q["answers"]]
            gt_frames = sorted({f for a in answers for f in tracks[a]["frame_ids"]})
            rows.append({
                "video_id": video_id,
                "question_id": str(q["id"]),
                "sample": f"{video_id}_q{q['id']}",
                "question": q["question"],
                "num_targets": len(answers),
                "target_bucket": "1" if len(answers) == 1 else "2+",
                "area": q.get("area", ""),
                "reasoning": q.get("reasoning", ""),
                "answer_track_ids": answers,
                "answer_labels": [tracks[a]["label"] for a in answers],
                "num_frames": int(meta["num_frames"]),
                "frame_rate": float(meta["frame_rate"]),
                "resolution": meta["resolution"],
                "duration_sec": int(meta["num_frames"]) / float(meta["frame_rate"]),
                "gt_frame_count": len(gt_frames),
                "is_camera_moving": meta.get("is_camera_moving"),
                "is_cup_game": meta.get("is_cup_game"),
            })

    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        strata[(row["target_bucket"], row["area"])].append(row)
    total = len(rows)
    keys = sorted(strata)
    exact = {k: args.size * len(strata[k]) / total for k in keys}
    alloc = {k: int(exact[k]) for k in keys}
    remainder = args.size - sum(alloc.values())
    for k in sorted(keys, key=lambda k: exact[k] - alloc[k], reverse=True)[:remainder]:
        alloc[k] += 1

    rng = random.Random(args.seed)
    chosen: list[dict] = []
    for k in keys:
        chosen.extend(rng.sample(strata[k], alloc[k]))
    chosen.sort(key=lambda r: (r["video_id"], int(r["question_id"])))

    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    args.tsv.write_text("".join(f"{r['video_id']}\t{r['question_id']}\n" for r in chosen), encoding="utf-8")
    summary = {
        "seed": args.seed,
        "size": len(chosen),
        "population": total,
        "strata": {f"{k[0]}|{k[1]}": {"population": len(strata[k]), "chosen": alloc[k]} for k in keys},
        "population_shares": {
            "target_bucket": dict(collections.Counter(r["target_bucket"] for r in rows)),
            "area": dict(collections.Counter(r["area"] for r in rows)),
            "reasoning": dict(collections.Counter(r["reasoning"] for r in rows)),
        },
        "subset_shares": {
            "target_bucket": dict(collections.Counter(r["target_bucket"] for r in chosen)),
            "area": dict(collections.Counter(r["area"] for r in chosen)),
            "reasoning": dict(collections.Counter(r["reasoning"] for r in chosen)),
        },
    }
    args.meta.write_text(json.dumps({"summary": summary, "questions": chosen}, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
