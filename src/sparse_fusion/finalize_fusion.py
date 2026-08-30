#!/usr/bin/env python3
"""Merge fusion shards and fill failed sparse samples from zero-shot dense tracks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sparse_common import load_json, load_subset, save_json


def add_question(
    merged: dict[str, Any],
    video_id: str,
    question_id: str,
    tracks: list[dict[str, Any]],
) -> None:
    questions = merged.setdefault(video_id, {"grounded_question": {}})["grounded_question"]
    if question_id in questions:
        raise ValueError(f"duplicate question {video_id}_q{question_id}")
    questions[question_id] = tracks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[str, Any] = {}
    for shard_path in sorted(args.fusion_root.glob("predictions_shard_*.json")):
        shard = load_json(shard_path)
        for video_id, video in shard.items():
            for question_id, tracks in video["grounded_question"].items():
                add_question(merged, video_id, str(question_id), tracks)

    fallback_samples: list[str] = []
    for video_id, question_id in load_subset(args.subset):
        existing = merged.get(video_id, {}).get("grounded_question", {}).get(question_id)
        if existing is not None:
            continue
        sample = f"{video_id}_q{question_id}"
        dense_path = args.dense_root / sample / "predictions.json"
        dense = load_json(dense_path)
        tracks = dense[video_id]["grounded_question"][question_id]
        add_question(merged, video_id, question_id, tracks)
        fallback_samples.append(sample)

    expected = {(video_id, question_id) for video_id, question_id in load_subset(args.subset)}
    actual = {
        (video_id, str(question_id))
        for video_id, video in merged.items()
        for question_id in video["grounded_question"]
    }
    if actual != expected:
        raise ValueError(
            f"coverage mismatch: missing={sorted(expected - actual)[:20]} "
            f"extra={sorted(actual - expected)[:20]}"
        )
    save_json(args.output, merged)
    save_json(args.summary, {
        "status": "complete",
        "num_questions": len(actual),
        "num_sparse_pipeline_v7_fused": len(actual) - len(fallback_samples),
        "num_pipeline_v7_fallback": len(fallback_samples),
        "pipeline_v7_fallback_samples": fallback_samples,
    })
    print(
        f"wrote {args.output}: questions={len(actual)} "
        f"dense_fallback={len(fallback_samples)}"
    )


if __name__ == "__main__":
    main()
