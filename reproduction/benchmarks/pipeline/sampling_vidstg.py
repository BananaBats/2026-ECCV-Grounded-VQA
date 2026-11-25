"""Exact native-clock sampling from the completed VidSTG Batch stage."""
import math

def sample_frames(num_frames, fps, sample_fps=1.):
    if num_frames < 1 or not math.isfinite(fps) or fps <= 0 or not 0 < sample_fps <= fps:
        raise ValueError('Invalid native frame count or FPS')
    output = []
    k = 0
    while True:
        frame = math.floor(k * fps / sample_fps + 0.5)
        if frame >= num_frames:
            break
        output.append(frame)
        k += 1
    assert output == sorted(set(output))
    return output
