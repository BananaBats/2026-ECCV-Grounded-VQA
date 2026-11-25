#!/usr/bin/env python3
"""Build label-free GVQA inputs and lossless videos from official SVAG sources."""
from __future__ import annotations

import argparse
import configparser
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build_inputs(annotation: Path, output: Path) -> dict:
    # Intentionally read only query text and identity fields. Training labels
    # (ID, Start, End) must never be made available to the inference pipeline.
    groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in json.loads(annotation.read_text()):
        key = (row["Dataset"], row["Video"])
        qid, query = str(row["QID"]), row["Language Query"].strip()
        previous = groups[key].setdefault(qid, query)
        if previous != query:
            raise ValueError(f"Conflicting query text for {key} QID={qid}")
    annotations, manifest, subset = {}, {}, []
    for index, ((dataset, name), queries) in enumerate(sorted(groups.items()), 1):
        video_id = f"video_{index}"
        annotations[video_id] = {"grounded_question": [
            {"id": qid, "question": queries[qid]}
            for qid in sorted(queries, key=int)
        ]}
        manifest[video_id] = {"dataset": dataset, "video_name": name}
        subset.extend(f"{video_id}\t{qid}\n" for qid in sorted(queries, key=int))
    write_json(output / "annotations/grounded_question_svag.json", annotations)
    write_json(output / "manifest.json", manifest)
    (output / "subset.tsv").write_text("".join(subset))
    (output / "smoke.tsv").write_text(subset[0])
    return manifest


def encode_frames(archive: ZipFile, members: list[str], output: Path, fps: float) -> dict:
    import cv2
    if not members:
        raise ValueError(f"No frames for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        # CRF 0 preserves source pixels. Use image2pipe to avoid a second copy
        # of hundreds of thousands of JPEG frames on the network volume.
        temporary = output.with_suffix(".partial.mp4")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "image2pipe", "-framerate", str(fps), "-vcodec", "mjpeg",
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                   "-crf", "0", "-pix_fmt", "yuvj444p", "-threads", "4", str(temporary)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            assert process.stdin is not None
            for name in members:
                process.stdin.write(archive.read(name))
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {output}")
        except BaseException:
            process.kill()
            process.wait()
            raise
        temporary.replace(output)
    capture = cv2.VideoCapture(str(output))
    metadata = {"frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": capture.get(cv2.CAP_PROP_FPS)}
    capture.release()
    if metadata["frames"] != len(members):
        raise ValueError(f"Frame count mismatch for {output}: {metadata}")
    return metadata


def materialize_videos(manifest: dict, downloads: Path, output: Path, ovis_fps: float) -> None:
    # OVIS images are an already sampled sequence. Do not insert/drop frames.
    # Fixed FPS controls playback only; SVAGEval uses sequence-frame indices.
    ovis_metadata = {}
    for split in ("train", "valid", "test"):
        path = downloads / f"OVIS_annotations_{split}.json"
        if path.exists():
            for video in json.loads(path.read_text())["videos"]:
                name = Path(video["file_names"][0]).parts[0]
                ovis_metadata[name] = (split, video)
    archives: dict[str, ZipFile] = {}
    video_info = {}
    try:
        for video_id, entry in manifest.items():
            dataset, name = entry["dataset"], entry["video_name"]
            if dataset == "OVIS":
                split, video = ovis_metadata[name]
                archive_name = f"OVIS_{split}.zip"
                fps = ovis_fps
            else:
                archive_name = f"{dataset}.zip"
            if archive_name not in archives:
                archives[archive_name] = ZipFile(downloads / archive_name)
            archive = archives[archive_name]
            names = archive.namelist()
            if dataset == "OVIS":
                candidates = [n for n in names if n.endswith("/" + video["file_names"][0])
                              or n == video["file_names"][0]]
                if len(candidates) != 1:
                    raise ValueError(f"Cannot locate unique OVIS sequence: {name}")
                prefix = candidates[0][:-len(video["file_names"][0])]
                members = [prefix + f for f in video["file_names"]]
            else:
                sequence = name + "-FRCNN" if dataset == "MOT17" else name
                ini_path = next(n for n in names if n.endswith(f"/{sequence}/seqinfo.ini"))
                config = configparser.ConfigParser()
                config.read_string(archive.read(ini_path).decode())
                fps = config.getfloat("Sequence", "frameRate")
                prefix = ini_path.removesuffix("seqinfo.ini") + "img1/"
                members = sorted(n for n in names if n.startswith(prefix) and n.endswith(".jpg"))
                if len(members) != config.getint("Sequence", "seqLength"):
                    raise ValueError(f"Source frame count mismatch: {sequence}")
            meta = encode_frames(archive, members, output / "videos" / f"{video_id}.mp4", fps)
            video_info[video_id] = {**entry, **meta}
            write_json(output / "video_info.json", video_info)
            print(f"{video_id} {dataset}/{name}: {meta['frames']} frames", flush=True)
    finally:
        for archive in archives.values():
            archive.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, default=Path("dataset/svag/downloads"))
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--ovis-fps", type=float, default=5)
    args = parser.parse_args()
    manifest = build_inputs(args.annotations, args.output)
    print(f"Prepared {len(manifest)} videos in {args.output}", flush=True)
    if args.videos:
        materialize_videos(manifest, args.downloads, args.output, args.ovis_fps)


if __name__ == "__main__":
    main()
