#!/usr/bin/env python3
"""Convert VidSTG sentences + VidOR videos into the pipeline's GVQA layout.

One VidSTG annotation entry is one (video, used_segment, relation) record with
one declarative caption and zero or more interrogative questions.  The model
input is the used_segment clip, so every distinct (vid, begin_fid, end_fid) is
cut into its own frame-exact clip and each sentence becomes one question on it.

Frame convention (checked against the released annotations and TubeDETR's
preprocessing): used_segment and temporal_gt are half-open [begin_fid, end_fid).
temporal_gt.end_fid can exceed used_segment.end_fid by one frame on the final
segment of a video; the tube is clipped to the evaluated segment.

Outputs under <project-root>/dataset/<split>/:
  annotations/grounded_question_<split>.json   model input (sentence text only)
  videos/<clip_id>.mp4                         frame-exact used_segment clips
  vidstg_clips.json                            clip offsets/sizes (no GT)
  subset.tsv                                   clip_id<TAB>question_id
  eval/vidstg_gt.json                          GT tubes; never read by the model

clip frame k == original frame (begin_fid + k).  Every evaluated frame id in
eval/ and in the exporter output is an ORIGINAL video frame id.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any, *, indent: int | None = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def clip_id_for(vid: str, begin_fid: int, end_fid: int) -> str:
    return f"{vid}_{int(begin_fid):05d}_{int(end_fid):05d}"


def question_id_for(entry_index: int, kind: str, index: int) -> str:
    """kind is 'c' (declarative caption) or 'q' (interrogative question)."""
    return f"{entry_index:05d}{kind}{index}"


def sample_name(clip_id: str, question_id: str) -> str:
    return f"{clip_id}_q{question_id}"


def flatten_sentences(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per sentence, with model inputs and eval-only fields separated."""
    sentences: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(entries):
        vid = str(entry["vid"])
        begin = int(entry["used_segment"]["begin_fid"])
        end = int(entry["used_segment"]["end_fid"])
        if end <= begin:
            raise ValueError(f"entry {entry_index} ({vid}) has an empty used_segment")
        tube_begin = int(entry["temporal_gt"]["begin_fid"])
        tube_end = min(int(entry["temporal_gt"]["end_fid"]), end)
        clip_id = clip_id_for(vid, begin, end)
        groups = (
            ("c", "declarative", entry.get("captions", [])),
            ("q", "interrogative", entry.get("questions", [])),
        )
        for kind, sentence_type, items in groups:
            for index, item in enumerate(items):
                question_id = question_id_for(entry_index, kind, index)
                sentences.append({
                    "sample": sample_name(clip_id, question_id),
                    "clip_id": clip_id,
                    "question_id": question_id,
                    "vid": vid,
                    "entry_index": entry_index,
                    "sentence_type": sentence_type,
                    "description": str(item["description"]).strip(),
                    # eval-only fields below
                    "target_type": item.get("type"),
                    "target_tid": int(item["target_id"]),
                    "eval_frames": [begin, end],
                    "tube": [max(tube_begin, begin), tube_end],
                })
    return sentences


def select_pilot(
    sentences: list[dict[str, Any]], per_type: int | None, seed: int
) -> list[dict[str, Any]]:
    if per_type is None:
        return sentences
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for sentence_type in ("declarative", "interrogative"):
        pool = [item for item in sentences if item["sentence_type"] == sentence_type]
        chosen.extend(rng.sample(pool, min(per_type, len(pool))))
    keep = {item["sample"] for item in chosen}
    return [item for item in sentences if item["sample"] in keep]


def index_files(root: Path, suffix: str) -> dict[str, Path]:
    """Map VidOR video id -> file under nested group folders (e.g. 0001/123.mp4)."""
    found: dict[str, Path] = {}
    for path in root.rglob(f"*{suffix}"):
        found.setdefault(path.stem, path)
    return found


def model_annotation(sentences: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The only sentence data the planner sees (load_question format)."""
    payload: dict[str, Any] = {}
    for item in sentences:
        video = payload.setdefault(item["clip_id"], {"grounded_question": []})
        video["grounded_question"].append({
            "id": item["question_id"],
            "question": item["description"],
            "sentence_type": item["sentence_type"],
        })
    return payload


def clip_metadata(
    entries: list[dict[str, Any]], sentences: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    clips: dict[str, Any] = {}
    for item in sentences:
        entry = entries[item["entry_index"]]
        begin, end = item["eval_frames"]
        clips.setdefault(item["clip_id"], {
            "vid": item["vid"],
            "begin_fid": begin,
            "end_fid": end,
            "num_frames": end - begin,
            "fps": float(entry["fps"]),
            "width": int(entry["width"]),
            "height": int(entry["height"]),
            "source_frame_count": int(entry["frame_count"]),
        })
    return clips


def trajectory_boxes(
    vidor_annotation: dict[str, Any], tid: int, frames: range
) -> dict[str, list[float]]:
    """Pixel xyxy boxes of one VidOR trajectory on the requested frames."""
    trajectories = vidor_annotation["trajectories"]
    boxes: dict[str, list[float]] = {}
    for frame in frames:
        if not 0 <= frame < len(trajectories):
            continue
        for item in trajectories[frame]:
            if int(item["tid"]) == tid:
                bbox = item["bbox"]
                boxes[str(frame)] = [
                    float(bbox["xmin"]), float(bbox["ymin"]),
                    float(bbox["xmax"]), float(bbox["ymax"]),
                ]
                break
    return boxes


def build_ground_truth(
    sentences: list[dict[str, Any]], vidor_annotations: dict[str, Path]
) -> tuple[dict[str, Any], list[str]]:
    by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sentences:
        by_vid[item["vid"]].append(item)
    gt: dict[str, Any] = {}
    warnings: list[str] = []
    for vid, items in sorted(by_vid.items()):
        path = vidor_annotations.get(vid)
        if path is None:
            raise FileNotFoundError(f"VidOR annotation for {vid} not found")
        annotation = load_json(path)
        for item in items:
            tube_begin, tube_end = item["tube"]
            boxes = trajectory_boxes(annotation, item["target_tid"], range(tube_begin, tube_end))
            missing = (tube_end - tube_begin) - len(boxes)
            if missing:
                warnings.append(f"{item['sample']}: {missing} tube frames without a GT box")
            gt[item["sample"]] = {
                "clip_id": item["clip_id"],
                "question_id": item["question_id"],
                "vid": vid,
                "sentence_type": item["sentence_type"],
                "target_type": item["target_type"],
                "target_tid": item["target_tid"],
                "eval_frames": item["eval_frames"],
                "tube": item["tube"],
                "boxes_xyxy": boxes,
                "width": int(annotation["width"]),
                "height": int(annotation["height"]),
            }
    return gt, warnings


def count_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
        ],
        check=True, text=True, capture_output=True,
    )
    return int(result.stdout.strip().split(",")[0])


def cut_clip(source: Path, destination: Path, begin: int, end: int, crf: int) -> None:
    """Frame-exact [begin, end) cut by decoder frame index, re-encoded to H.264."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.mp4")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", f"select=between(n\\,{begin}\\,{end - 1}),setpts=N/FRAME_RATE/TB,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-vsync", "passthrough", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", str(crf), "-pix_fmt", "yuv420p", str(tmp),
        ],
        check=True, capture_output=True, text=True,
    )
    os.replace(tmp, destination)


def write_clips(
    clips: dict[str, Any],
    videos: dict[str, Path],
    video_dir: Path,
    *,
    crf: int,
    verify: bool,
) -> list[str]:
    problems: list[str] = []
    for position, (clip_id, meta) in enumerate(sorted(clips.items()), start=1):
        destination = video_dir / f"{clip_id}.mp4"
        if destination.is_file() and not verify:
            continue
        source = videos.get(meta["vid"])
        if source is None:
            problems.append(f"{clip_id}: source video {meta['vid']} not found")
            continue
        if not destination.is_file():
            print(f"[{position}/{len(clips)}] cut {clip_id}", flush=True)
            try:
                cut_clip(source, destination, meta["begin_fid"], meta["end_fid"], crf)
            except subprocess.CalledProcessError as exc:
                problems.append(f"{clip_id}: ffmpeg failed: {exc.stderr.strip()[-300:]}")
                continue
        if verify:
            decoded = count_frames(destination)
            meta["decoded_frames"] = decoded
            if decoded != meta["num_frames"]:
                problems.append(
                    f"{clip_id}: decoded {decoded} frames, expected {meta['num_frames']}"
                )
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vidstg-annotations", type=Path, required=True,
                        help="VidSTG {split}_annotations.json")
    parser.add_argument("--vidor-video-root", type=Path,
                        help="VidOR video/ folder for this split (group/<vid>.mp4)")
    parser.add_argument("--vidor-annotation-root", type=Path,
                        help="VidOR annotation/<training|validation>/ folder (group/<vid>.json)")
    parser.add_argument("--project-root", type=Path,
                        default=Path(os.environ.get("VQA_PROJECT_ROOT", ".")))
    parser.add_argument("--split-name", default="vidstg_test")
    parser.add_argument("--pilot-per-type", type=int,
                        help="keep N declarative + N interrogative sentences")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-clips", action="store_true")
    parser.add_argument("--skip-gt", action="store_true")
    parser.add_argument("--verify-clips", action="store_true",
                        help="count decoded frames of every clip (slow)")
    parser.add_argument("--crf", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_json(args.vidstg_annotations)
    sentences = select_pilot(flatten_sentences(entries), args.pilot_per_type, args.seed)
    # Group by clip so persistent workers reuse decoded frames.
    sentences.sort(key=lambda item: (item["clip_id"], item["question_id"]))
    split_root = args.project_root.resolve() / "dataset" / args.split_name

    save_json(
        split_root / "annotations" / f"grounded_question_{args.split_name}.json",
        model_annotation(sentences),
    )
    clips = clip_metadata(entries, sentences)
    subset = split_root / "subset.tsv"
    subset.write_text(
        "".join(f"{item['clip_id']}\t{item['question_id']}\n" for item in sentences),
        encoding="utf-8",
    )
    counts = {
        sentence_type: sum(item["sentence_type"] == sentence_type for item in sentences)
        for sentence_type in ("declarative", "interrogative")
    }
    print(f"sentences={len(sentences)} {counts} clips={len(clips)} subset={subset}")

    status = 0
    if not args.skip_clips:
        if args.vidor_video_root is None:
            raise SystemExit("--vidor-video-root is required unless --skip-clips")
        problems = write_clips(
            clips, index_files(args.vidor_video_root, ".mp4"), split_root / "videos",
            crf=args.crf, verify=args.verify_clips,
        )
        for problem in problems:
            print(f"CLIP PROBLEM {problem}", file=sys.stderr)
        status = 1 if problems else status
    save_json(split_root / "vidstg_clips.json", clips)

    if not args.skip_gt:
        if args.vidor_annotation_root is None:
            raise SystemExit("--vidor-annotation-root is required unless --skip-gt")
        gt, warnings = build_ground_truth(
            sentences, index_files(args.vidor_annotation_root, ".json")
        )
        save_json(split_root / "eval" / "vidstg_gt.json", gt, indent=None)
        for warning in warnings[:20]:
            print(f"GT WARNING {warning}", file=sys.stderr)
        print(f"gt samples={len(gt)} warnings={len(warnings)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
