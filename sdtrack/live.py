"""Produce model observations for the same fusion function used in replay."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from google import genai
from google.genai import types
from sam3.model_builder import build_sam3_video_predictor

from .media import video_info
from .planning import plan_targets
from .sparse import localize
from .tracking import dense_tracks


def infer(video, question, project, model, checkpoint, target_kind="query"):
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 requires a CUDA GPU")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg must be on PATH for sparse frame decoding")
    info = video_info(video)
    with TemporaryDirectory() as scratch, genai.Client(
        vertexai=True, project=project, location="global",
        http_options=types.HttpOptions(api_version="v1", timeout=300000),
    ) as client:
        work = Path(scratch)
        print("Planning targets and anchor boxes with Gemini...", flush=True)
        plan = plan_targets(client, model, video, question, info, work, target_kind)
        print("Tracking the planned identities with SAM3...", flush=True)
        predictor = build_sam3_video_predictor(checkpoint_path=str(checkpoint), gpus_to_use=[0])
        try:
            with torch.inference_mode():
                sam = dense_tracks(predictor, video, plan, info)
        finally:
            shutdown = getattr(predictor, "shutdown", None)
            if shutdown:
                shutdown()
        print("Localizing the same identities at sparse frames with Gemini...", flush=True)
        gemini = localize(client, model, video, plan, info, work)
    return plan, sam, gemini
