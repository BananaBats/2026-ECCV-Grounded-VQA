#!/usr/bin/env python3
"""Shared, dependency-light helpers for sparse Gemini grounding and fusion."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parents[1]
PROJECT_ROOT = Path(os.environ.get("VQA_PROJECT_ROOT", str(RELEASE_ROOT))).resolve()

DEFAULT_SUBSET = RELEASE_ROOT / "config" / "test.tsv"
DEFAULT_PLAN_ROOT = RELEASE_ROOT / "plans" / "test"
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "dataset" / "test" / "videos"
DEFAULT_DENSE_ROOT = RELEASE_ROOT / "outputs" / "dense"
DEFAULT_SPARSE_ROOT = RELEASE_ROOT / "outputs" / "sparse"
DEFAULT_FUSED_ROOT = RELEASE_ROOT / "outputs" / "fused"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so a preempted pod never leaves a fake result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def load_subset(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{lineno}: expected video_id<TAB>question_id")
        rows.append((fields[0].strip(), fields[1].strip()))
    return rows


def sample_name(video_id: str, question_id: str | int) -> str:
    return f"{video_id}_q{question_id}"


def select_rows(
    rows: Iterable[tuple[str, str]],
    *,
    samples: set[str] | None,
    num_shards: int,
    shard_index: int,
    limit: int | None,
) -> list[tuple[str, str]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    chosen = [
        row for index, row in enumerate(rows)
        if index % num_shards == shard_index
        and (samples is None or sample_name(*row) in samples)
    ]
    return chosen[:limit] if limit is not None else chosen


def sane_yxyx(box: Iterable[float], *, min_size: float = 1.0) -> list[float]:
    values = [float(value) for value in box]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid box: {values}")
    y0, x0, y1, x1 = (min(1000.0, max(0.0, value)) for value in values)
    if y1 - y0 < min_size or x1 - x0 < min_size:
        raise ValueError(f"degenerate yxyx box: {values}")
    return [y0, x0, y1, x1]


def yxyx_1000_to_xyxy_normalized(box: Iterable[float]) -> list[float]:
    y0, x0, y1, x1 = sane_yxyx(box)
    return [x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0]


def xyxy_normalized_to_yxyx_1000(box: Iterable[float]) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box]
    return sane_yxyx([y0 * 1000.0, x0 * 1000.0, y1 * 1000.0, x1 * 1000.0])


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ay0, ax0, ay1, ax1 = [float(value) for value in a]
    by0, bx0, by1, bx1 = [float(value) for value in b]
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    intersection = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def plan_target_ids(plan: dict[str, Any]) -> list[int]:
    ids = [int(target["obj_id"]) for target in plan.get("targets", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("plan has duplicate obj_id values")
    return ids


def target_is_visibly_confirmed(target: dict[str, Any], frame_idx: int) -> bool:
    segments = target.get("visibility_segments") or []
    if not segments:
        return True
    return any(
        int(segment["first_frame_idx"]) <= frame_idx <= int(segment["last_frame_idx"])
        for segment in segments
    )


def expected_ids(plan: dict[str, Any], frame_idx: int) -> set[int]:
    """IDs guaranteed visible by the saved plan at this exact frame.

    The source plan calls these *visibility* segments, not presence spans.
    They must never be used to forbid a target during an occlusion gap.
    """
    return {
        int(target["obj_id"])
        for target in plan.get("targets", [])
        if target_is_visibly_confirmed(target, frame_idx)
    }


def compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove old API metadata and local paths before sending the plan again."""
    return {
        "answer_noun_phrase": plan.get("answer_noun_phrase", ""),
        "decision_summary": plan.get("decision_summary", ""),
        "first_visible_frame_idx": plan.get("first_visible_frame_idx"),
        "last_visible_frame_idx": plan.get("last_visible_frame_idx"),
        "visibility_boundary_reason": plan.get("visibility_boundary_reason", ""),
        "targets": [
            {
                "object_id": int(target["obj_id"]),
                "object_label": target.get("object_label", ""),
                "visual_identity_description": target.get("visual_identity_description", ""),
                "visibility_segments": target.get("visibility_segments", []),
            }
            for target in plan.get("targets", [])
        ],
        "anchor_boxes": [
            {
                "frame_idx": int(anchor["frame_idx"]),
                "boxes": [
                    {
                        "object_id": int(box["obj_id"]),
                        "box_2d_yxyx_1000": box["box_2d"],
                    }
                    for box in anchor.get("boxes", [])
                ],
            }
            for anchor in plan.get("anchors", [])
        ],
    }
