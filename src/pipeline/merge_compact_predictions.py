#!/usr/bin/env python3
"""Merge per-sample compact pipeline-v7 boxes into one GVQA submission."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        video_id, question_id = line.split("\t")
        rows.append((video_id.strip(), question_id.strip()))
    return rows


def validate_tracks(tracks: Any, sample: str) -> None:
    if not isinstance(tracks, list):
        raise TypeError(f"{sample}: tracks must be a list")
    for track in tracks:
        frame_ids = track.get("frame_ids")
        boxes = track.get("bounding_boxes")
        if not isinstance(frame_ids, list) or not isinstance(boxes, list):
            raise TypeError(f"{sample}: malformed track")
        if len(frame_ids) != len(boxes) or not frame_ids:
            raise ValueError(f"{sample}: frame/box length mismatch or empty track")
        if frame_ids != list(range(len(frame_ids))):
            raise ValueError(f"{sample}: compact track is not dense from frame zero")
        for box in boxes:
            if not (
                isinstance(box, list)
                and len(box) == 4
                and 0 <= box[0] < box[2] <= 1
                and 0 <= box[1] < box[3] <= 1
            ):
                raise ValueError(f"{sample}: invalid normalized XYXY box {box}")


def load_sample(root: Path, video_id: str, question_id: str) -> list[dict[str, Any]]:
    sample = f"{video_id}_q{question_id}"
    path = root / sample / "predictions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    tracks = value[video_id]["grounded_question"][question_id]
    validate_tracks(tracks, sample)
    return tracks


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    submission: dict[str, Any] = {}
    missing: list[str] = []
    invalid: list[str] = []
    rows = read_rows(args.subset)
    for video_id, question_id in rows:
        sample = f"{video_id}_q{question_id}"
        try:
            tracks = load_sample(args.output_root, video_id, question_id)
        except FileNotFoundError:
            missing.append(sample)
            continue
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            invalid.append(f"{sample}: {error}")
            continue
        submission.setdefault(video_id, {"grounded_question": {}})
        submission[video_id]["grounded_question"][question_id] = tracks

    if invalid:
        raise RuntimeError("invalid compact samples:\n" + "\n".join(invalid[:20]))
    if missing and not args.allow_missing:
        raise RuntimeError(
            f"refusing incomplete submission: {len(missing)}/{len(rows)} missing; "
            f"first={missing[:10]}"
        )
    atomic_json(args.output, submission)
    completed = len(rows) - len(missing)
    print(f"wrote {args.output}: completed={completed}/{len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
