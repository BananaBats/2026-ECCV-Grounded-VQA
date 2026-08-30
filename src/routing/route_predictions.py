#!/usr/bin/env python3
"""Apply the validation-selected question router used for the final submission."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


APPEAR_TWICE = "Track the objects shown to the camera more than once."


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    os.replace(temporary, path)


def read_subset(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected video_id<TAB>question_id")
        rows.append((fields[0].strip(), fields[1].strip()))
    return rows


def merge_prediction_shards(root: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    shard_paths = sorted(root.glob("predictions_shard_*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"no predictions_shard_*.json files under {root}")
    for shard_path in shard_paths:
        shard = load_json(shard_path)
        for video_id, payload in shard.items():
            target = merged.setdefault(video_id, {"grounded_question": {}})
            questions = payload.get("grounded_question", {})
            overlap = set(target["grounded_question"]) & set(questions)
            if overlap:
                raise ValueError(f"duplicate questions in shards for {video_id}: {sorted(overlap)}")
            target["grounded_question"].update(questions)
    return merged


def load_predictions(path: Path) -> dict[str, Any]:
    return merge_prediction_shards(path) if path.is_dir() else load_json(path)


def is_alternate_route(plan: dict[str, Any]) -> bool:
    question = str(plan.get("question", ""))
    return (
        len(plan.get("targets", [])) > 1
        and "occlusion game" not in question.lower()
        and question.strip() != APPEAR_TWICE
    )


def route(
    base: dict[str, Any],
    alternate: dict[str, Any],
    rows: list[tuple[str, str]],
    plan_root: Path,
    *,
    allow_missing_alternate: bool,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    routed: list[str] = []
    kept: list[str] = []
    unavailable: list[str] = []
    for video_id, question_id in rows:
        sample = f"{video_id}_q{question_id}"
        plan = load_json(plan_root / sample / "multi_anchor_plan.json")
        eligible = is_alternate_route(plan)
        replacement = (
            alternate.get(video_id, {})
            .get("grounded_question", {})
            .get(question_id)
        )
        if eligible and replacement is not None:
            base[video_id]["grounded_question"][question_id] = replacement
            routed.append(sample)
        elif eligible:
            unavailable.append(sample)
            kept.append(sample)
        else:
            kept.append(sample)
    if unavailable and not allow_missing_alternate:
        raise RuntimeError(
            f"alternate prediction is missing for {len(unavailable)} eligible samples; "
            f"first={unavailable[:10]}"
        )
    return base, routed, kept, unavailable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--alternate",
        type=Path,
        required=True,
        help="monolithic predictions JSON or a directory of prediction shards",
    )
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-missing-alternate", action="store_true")
    args = parser.parse_args()

    routed_output, routed, kept, unavailable = route(
        load_predictions(args.base),
        load_predictions(args.alternate),
        read_subset(args.subset),
        args.plans,
        allow_missing_alternate=args.allow_missing_alternate,
    )
    atomic_json(args.output, routed_output)
    manifest = {
        "base": str(args.base),
        "alternate": str(args.alternate),
        "rule": "plan has >1 target and question is neither occlusion-game nor appear-twice",
        "num_routed": len(routed),
        "num_kept_base": len(kept),
        "eligible_but_unavailable": unavailable,
        "routed_samples": routed,
    }
    atomic_json(args.manifest, manifest, indent=2)
    print(
        f"wrote {args.output}: routed={len(routed)} kept_base={len(kept)} "
        f"unavailable={len(unavailable)}"
    )


if __name__ == "__main__":
    main()
