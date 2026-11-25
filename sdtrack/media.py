"""Read zero-based source frames and make the labeled video used for planning."""
from pathlib import Path

import cv2
import numpy as np


def video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        info = {
            "num_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": capture.get(cv2.CAP_PROP_FPS),
        }
    finally:
        capture.release()
    if not all(value > 0 for value in info.values()):
        raise ValueError(f"Could not read video metadata from {path}")
    return info


def read_frames(path, indices):
    """Yield (source index, BGR image) in decode order."""
    wanted = set(indices)
    if not wanted:
        return
    capture = cv2.VideoCapture(str(path))
    try:
        for index in range(max(wanted) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video decode stopped at source frame {index}")
            if index in wanted:
                yield index, frame
    finally:
        capture.release()


def labeled_proxy(video, destination, info):
    count = info["num_frames"]
    indices = np.linspace(0, count - 1, min(count, 192), dtype=np.int64)
    scale = min(1.0, 512 / max(info["width"], info["height"]))
    size = tuple(max(1, round(info[key] * scale)) for key in ("width", "height"))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             min(max(info["fps"], 1.0), 6.0), size)
    try:
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not encode the planning video")
        for index, frame in read_frames(video, indices):
            if scale < 1:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            label = f"orig_frame={index}"
            for color, width in [((0, 0, 0), 4), ((255, 255, 255), 2)]:
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, color, width, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return destination
