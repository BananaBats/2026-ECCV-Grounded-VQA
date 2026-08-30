#!/usr/bin/env python3
"""Validate the structural and numeric invariants of a GVQA submission JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def validate(data: dict[str, Any]) -> dict[str, int]:
    questions = tracks = boxes = 0
    errors: list[str] = []
    for video_id, video in data.items():
        for question_id, items in video.get("grounded_question", {}).items():
            sample = f"{video_id}_q{question_id}"
            questions += 1
            if not isinstance(items, list) or len(items) > 10:
                errors.append(f"{sample}: invalid track list")
                continue
            ids: list[int] = []
            for track in items:
                tracks += 1
                ids.append(int(track["id"]))
                frame_ids = track["frame_ids"]
                coordinates = track["bounding_boxes"]
                if frame_ids != sorted(set(frame_ids)):
                    errors.append(f"{sample}: frame_ids are not unique and sorted")
                if len(frame_ids) != len(coordinates):
                    errors.append(f"{sample}: frame/box length mismatch")
                    continue
                for box in coordinates:
                    boxes += 1
                    valid = (
                        isinstance(box, list)
                        and len(box) == 4
                        and all(math.isfinite(float(value)) for value in box)
                        and 0 <= float(box[0]) < float(box[2]) <= 1
                        and 0 <= float(box[1]) < float(box[3]) <= 1
                    )
                    if not valid:
                        errors.append(f"{sample}: invalid normalized XYXY box {box}")
                        break
            if len(ids) != len(set(ids)):
                errors.append(f"{sample}: duplicate track IDs")
    if errors:
        raise ValueError(f"submission has {len(errors)} error(s); first={errors[:20]}")
    return {"videos": len(data), "questions": questions, "tracks": tracks, "boxes": boxes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--expected-questions", type=int)
    args = parser.parse_args()
    with args.predictions.open("r", encoding="utf-8") as handle:
        summary = validate(json.load(handle))
    if args.expected_questions is not None and summary["questions"] != args.expected_questions:
        raise ValueError(
            f"expected {args.expected_questions} questions, got {summary['questions']}"
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
