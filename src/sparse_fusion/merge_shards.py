#!/usr/bin/env python3
"""Merge completed fusion shard JSON files into one submission candidate."""
from __future__ import annotations

import argparse
from pathlib import Path

from sparse_common import DEFAULT_FUSED_ROOT, DEFAULT_SUBSET, load_json, load_subset, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.input_root = args.input_root.resolve()
    output = (args.output or (args.input_root / "predictions.json")).resolve()
    merged: dict = {}
    for path in sorted(args.input_root.glob("predictions_shard_*.json")):
        for video_id, payload in load_json(path).items():
            destination = merged.setdefault(video_id, {"grounded_question": {}})
            questions = payload["grounded_question"]
            duplicate = set(destination["grounded_question"]) & set(questions)
            if duplicate:
                raise ValueError(f"duplicate {video_id} questions: {sorted(duplicate)}")
            destination["grounded_question"].update(questions)
    expected = {(video_id, str(question_id)) for video_id, question_id in load_subset(args.subset)}
    actual = {
        (video_id, str(question_id))
        for video_id, payload in merged.items()
        for question_id in payload["grounded_question"]
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"submission coverage mismatch: missing={missing[:20]} extra={extra[:20]}")
    save_json(output, merged)
    print(f"merged {len(actual)} questions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
