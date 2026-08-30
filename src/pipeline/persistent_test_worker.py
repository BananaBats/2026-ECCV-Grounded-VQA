#!/usr/bin/env python3
"""Persistent single-GPU worker for compact pipeline-v7 test replay."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
import zlib
from pathlib import Path
from typing import Any

import torch

import merge_compact_predictions as merger
import multi_anchor_box_baseline as pipeline


def read_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        video_id, question_id = line.split("\t")
        rows.append((video_id.strip(), question_id.strip()))
    return rows


def video_shard(video_id: str, num_shards: int) -> int:
    return zlib.adler32(video_id.encode("utf-8")) % num_shards


def sample_complete(
    root: Path,
    video_id: str,
    question_id: str,
    *,
    session_mode: str,
    sam_stride: int,
) -> bool:
    run_dir = root / f"{video_id}_q{question_id}"
    try:
        result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
        if result.get("status") != "complete" or not result.get("compact_prediction_path"):
            return False
        if result.get("session_mode", "joint") != session_mode:
            return False
        if int(result.get("sam_stride", 3)) != sam_stride:
            return False
        merger.load_sample(root, video_id, question_id)
        return not any(run_dir.glob("*.pkl"))
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def sample_args(
    video_id: str,
    question_id: str,
    plan_path: Path,
    output_root: Path,
    project: str,
    *,
    session_mode: str,
    sam_stride: int,
    boundary_radius: int,
    segment_gate_margin: int,
    propagation_start: str,
) -> argparse.Namespace:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return pipeline.parse_args(
        [
            "--video-id", video_id,
            "--question-id", question_id,
            "--reuse-plan-json", str(plan_path),
            "--max-anchor-frames", str(plan["max_anchor_frames"]),
            "--sam-stride", str(sam_stride),
            "--boundary-radius", str(boundary_radius),
            "--segment-gate-margin", str(segment_gate_margin),
            "--propagation-start", propagation_start,
            "--session-mode", session_mode,
            "--no-retain-ungated-baseline",
            "--no-render-visualizations",
            "--compact-box-output",
            "--google-cloud-project", project,
            "--output-root", str(output_root),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=int, default=15)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--session-mode", choices=("joint", "per-object"), default="joint"
    )
    parser.add_argument("--sam-stride", type=int, default=3)
    parser.add_argument("--boundary-radius", type=int, default=15)
    parser.add_argument("--segment-gate-margin", type=int, default=15)
    parser.add_argument(
        "--propagation-start", choices=("first", "middle", "last"), default="middle"
    )
    args = parser.parse_args()
    if not 0 <= args.rank < args.num_shards:
        parser.error("--rank must be in [0, num-shards)")
    if args.sam_stride < 1:
        parser.error("--sam-stride must be at least 1")
    if args.boundary_radius < 0:
        parser.error("--boundary-radius must be non-negative")

    rows = [
        row for row in read_rows(args.subset)
        if video_shard(row[0], args.num_shards) == args.rank
    ]
    pending = [
        row
        for row in rows
        if not sample_complete(
            args.output_root,
            *row,
            session_mode=args.session_mode,
            sam_stride=args.sam_stride,
        )
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    resumed = len(rows) - len(pending)
    print(
        f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
        f"persistent shard={args.rank} assigned={len(rows)} pending={len(pending)} "
        f"already_complete={resumed} session_mode={args.session_mode} "
        f"sam_stride={args.sam_stride}",
        flush=True,
    )
    if not pending:
        return

    started = time.monotonic()
    predictor = pipeline.base.build_sam3_video_predictor(
        gpus_to_use=[torch.cuda.current_device()]
    )
    print(
        f"persistent shard={args.rank}: model ready in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    frames_cache: dict[str, tuple[list[Any], dict[str, Any]]] = {}
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "unused-for-saved-plan-replay")
    completed = 0
    failed = 0

    for position, (video_id, question_id) in enumerate(pending, start=1):
        sample = f"{video_id}_q{question_id}"
        ok = False
        for attempt in range(1, args.max_attempts + 1):
            sample_started = time.monotonic()
            print(
                f"[{position}/{len(pending)}] {sample}: attempt={attempt}/{args.max_attempts}",
                flush=True,
            )
            try:
                parsed = sample_args(
                    video_id,
                    question_id,
                    args.plan_root / sample / "multi_anchor_plan.json",
                    args.output_root,
                    project,
                    session_mode=args.session_mode,
                    sam_stride=args.sam_stride,
                    boundary_radius=args.boundary_radius,
                    segment_gate_margin=args.segment_gate_margin,
                    propagation_start=args.propagation_start,
                )
                pipeline.run(parsed, predictor=predictor, frames_cache=frames_cache)
                if not sample_complete(
                    args.output_root,
                    video_id,
                    question_id,
                    session_mode=args.session_mode,
                    sam_stride=args.sam_stride,
                ):
                    raise RuntimeError("compact completion validation failed")
                elapsed = time.monotonic() - sample_started
                completed += 1
                ok = True
                print(
                    f"[{position}/{len(pending)}] {sample}: complete {elapsed:.1f}s",
                    flush=True,
                )
                break
            except Exception as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                    "out of memory" in str(error).lower()
                )
                kind = "CUDA_OOM" if is_oom else type(error).__name__
                print(
                    f"{sample}: attempt={attempt} failed kind={kind}",
                    flush=True,
                )
                traceback.print_exc()
                frames_cache.clear()
                gc.collect()
                torch.cuda.empty_cache()
                if attempt < args.max_attempts:
                    delay = args.retry_seconds * attempt
                    if is_oom:
                        # De-synchronize the two workers sharing one physical GPU.
                        delay = max(30, delay) + (args.rank * 3)
                    print(f"{sample}: retrying in {delay}s", flush=True)
                    time.sleep(delay)
        if not ok:
            failed += 1
        elif completed % 25 == 0:
            gc.collect()

    print(
        f"persistent shard={args.rank}: finished completed={completed} "
        f"resumed={resumed} failed={failed}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
