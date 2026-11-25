"""Replay the paper example, or run its model stages on a video and question."""
import argparse
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from sdtrack.fusion import fuse, sane_yxyx


EXAMPLE = Path(__file__).resolve().parent / "examples" / "remote_control"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unpack_tracks(records):
    return {
        int(identity): {int(frame): sane_yxyx(box) for frame, box in track.items()}
        for identity, track in records.items()
    }


def check_reference(tracks, expected):
    if len(tracks) != len(expected):
        raise ValueError("Object count differs from the archived prediction")
    for track, reference in zip(tracks, expected):
        if track["id"] != reference["id"] or track["frame_ids"] != reference["frame_ids"]:
            raise ValueError("Object IDs or frame indices differ from the archived prediction")
        if len(track["bounding_boxes"]) != len(reference["bounding_boxes"]):
            raise ValueError("Box count differs from the archived prediction")
        for box, other in zip(track["bounding_boxes"], reference["bounding_boxes"]):
            if len(box) != 4 or len(other) != 4 or not all(
                math.isclose(a, b, rel_tol=0, abs_tol=1e-9) for a, b in zip(box, other)
            ):
                raise ValueError(f"Box differs from the archived prediction: {box} vs {other}")


def save_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8")
    temp_path = Path(temp.name)
    try:
        with temp:
            json.dump(payload, temp, allow_nan=False, indent=2)
            temp.write("\n")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run Gemini and SAM3")
    parser.add_argument("--project", help="Vertex AI project for live inference")
    parser.add_argument("--checkpoint", type=Path, help="local SAM3 checkpoint for live inference")
    parser.add_argument("--model", help="Gemini model for live inference (default: gemini-3.7-flash)")
    parser.add_argument("--video", type=Path, help="use a different video with --live and --question")
    parser.add_argument("--question", help="tracking question for --video")
    parser.add_argument("--target-kind", choices=("query", "person", "object"),
                        help="live target selection (default: query for a new clip, object for the example)")
    parser.add_argument("--output", type=Path, help="destination for the prediction JSON")
    args = parser.parse_args()
    if not args.live and any((args.video, args.question, args.project, args.checkpoint,
                              args.model, args.target_kind)):
        parser.error("model and video options require --live")
    if bool(args.video) != (args.question is not None) or (args.question is not None and not args.question.strip()):
        parser.error("supply --video and a nonempty --question together")
    if args.video:
        video = args.video.resolve()
        query = {"video_id": video.stem, "question_id": "0", "question": args.question.strip()}
    else:
        query = read_json(EXAMPLE / "query.json")
        video = EXAMPLE / (query["video_id"] + ".mp4")

    output = args.output or Path("outputs") / ("live_tracks.json" if args.live else "tracks.json")
    if EXAMPLE in output.resolve().parents or output.resolve() in (EXAMPLE, video):
        parser.error("choose an output path outside the example files and the input video")
    print(query["question"], flush=True)
    if args.live:
        project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project or not args.checkpoint:
            parser.error("--live requires a Vertex AI project and --checkpoint")
        try:
            from sdtrack.live import infer
        except ImportError as exc:
            parser.exit(1, f"Live inference dependency unavailable: {exc}. See README.md.\n")
        target_kind = args.target_kind or ("query" if args.video else "object")
        plan, sam, gemini = infer(video, query["question"], project,
                                  args.model or "gemini-3.7-flash", args.checkpoint, target_kind)
    else:
        plan = read_json(EXAMPLE / "plan.json")
        saved = read_json(EXAMPLE / "observations.json")
        sam, gemini = unpack_tracks(saved["sam3"]), unpack_tracks(saved["gemini"])

    result = fuse(query, plan, sam, gemini)
    tracks = result[query["video_id"]]["grounded_question"][str(query["question_id"])]
    if not args.live:
        expected = read_json(EXAMPLE / "expected_tracks.json")
        check_reference(tracks, expected[query["video_id"]]["grounded_question"][str(query["question_id"])])
        print("Matched the archived object IDs, frame indices, and box coordinates.")
    save_result(output, result)
    count = sum(len(track["frame_ids"]) for track in tracks)
    noun = "track" if len(tracks) == 1 else "tracks"
    print(f"{len(tracks)} {noun}, {count} boxes. Saved to {output}")


if __name__ == "__main__":
    main()
