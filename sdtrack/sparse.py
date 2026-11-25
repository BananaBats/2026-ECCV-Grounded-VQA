"""Gemini amodal localization on every thirtieth source frame."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Literal

from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SparseObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: int = Field(ge=0, le=31)
    visibility: Literal["visible", "partially_occluded", "fully_occluded"]
    box_2d: list[int] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("box_2d")
    @classmethod
    def valid_box(cls, value: list[int]) -> list[int]:
        values = [int(item) for item in value]
        if any(item < 0 or item > 1000 for item in values):
            raise ValueError("box_2d values must be in 0..1000")
        y0, x0, y1, x1 = values
        if y1 <= y0 or x1 <= x0:
            raise ValueError("box_2d must have positive area")
        return values


class SparseFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    objects: list[SparseObject] = Field(max_length=8)


class SparseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames: list[SparseFrame] = Field(min_length=1)


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
    """Format identities, visibility intervals and anchor boxes for the sparse prompt."""
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


STAGE2_V2_RULES = """
Visibility calibration: use partially_occluded only when a substantial part of
an object is hidden behind another scene object. An object merely held or
touched by a hand is visible. For partially_occluded objects keep the box tight
to the object's actual physical extent; extend it only over the hidden part you
can confidently infer, and never inflate it beyond the object's true size.
""".strip()


def build_prompt(plan: dict[str, Any], frame_indices: list[int], video: dict[str, Any]) -> str:
    plan_json = json.dumps(compact_plan(plan), ensure_ascii=False)
    visible_required = {
        str(frame): sorted(expected_ids(plan, frame)) for frame in frame_indices
    }
    known_ids = sorted(int(target["obj_id"]) for target in plan.get("targets", []))
    indices = ", ".join(str(item) for item in frame_indices)
    return f"""
You are Stage 2. The attached images are decoder-exact source frames in the listed
order: {indices}. Use the following immutable saved target plan; retain its IDs and
never create a new ID:
{plan_json}

For every frame and every target, first identify that target's own visual evidence,
expected physical size, and position from its before/after motion. Then return exactly
one frames record per listed frame_idx and exactly one amodal box_2d
[ymin, xmin, ymax, xmax] normalized 0..1000 plus visibility for every target
physically in-frame.

The only known target IDs are {known_ids}. The saved visibility segments mean
"visibly observed", NOT "physically present". IDs visibly confirmed at each requested
frame are {json.dumps(visible_required)} and must be returned there. A gap between
visible segments is a high-priority occlusion interval: if the same physical object
remains in the scene, return it as partially_occluded or fully_occluded. If it truly
left the scene, omit it. Never use visibility-segment membership to discard a hidden
object. An omitted object means physically absent, not merely invisible.

HARD PROHIBITIONS:
- A box must cover exactly one target identity. Never return a union, pile, word,
  collection, group, shared occlusion region, or any other multi-object box.
- Never copy the same box to two different object_id values. If distinct targets overlap
  or are both occluded, infer a separate full amodal box for each physical instance.
- An occluder is not the target: never return the occluder's whole box, its visible
  fragment, or a target-plus-occluder box. For example, if a small target is hidden
  under a cup, return the inferred small target box, not the cup box.

Use partially_occluded only when some pixels of that specific target are visible, and
box its complete inferred extent rather than the visible fragment. Use fully_occluded
only when zero pixels of that specific target are visible and temporal evidence supports
that it remains in-frame; infer its own location, scale, and full extent from context.
Do not use a large occluder or group box as a substitute for this inference.

Before returning JSON, audit every frame: each object_id must have a distinct physical
instance; each box must match that instance's scale; no box may include another target
or its occluder; and no two distinct IDs may share identical coordinates. The source
resolution is {video['width']}x{video['height']}. These are fixed 30-frame-grid
positions.

{STAGE2_V2_RULES}
""".strip()


def build_contents(types: Any, prompt: str, frame_indices: list[int], frame_dir: Path) -> list[Any]:
    parts = [types.Part.from_text(text=prompt)]
    parts.extend(
        types.Part.from_bytes(
            data=(frame_dir / f"frame_{frame_idx:06d}.jpg").read_bytes(),
            mime_type="image/jpeg",
        )
        for frame_idx in frame_indices
    )
    return [types.Content(role="user", parts=parts)]


def validate_response(
    response: SparseResponse,
    plan: dict[str, Any],
    requested: list[int],
) -> None:
    returned = [frame.frame_idx for frame in response.frames]
    if returned != requested:
        raise ValueError(f"frame sequence mismatch: expected={requested}, returned={returned}")
    known = {int(target["obj_id"]) for target in plan.get("targets", [])}
    for frame in response.frames:
        ids = [item.object_id for item in frame.objects]
        required = expected_ids(plan, frame.frame_idx)
        if len(ids) != len(set(ids)):
            raise ValueError(f"frame {frame.frame_idx} has duplicate object IDs: {ids}")
        if not set(ids) <= known:
            raise ValueError(
                f"frame {frame.frame_idx} has unknown IDs: returned={ids}, known={sorted(known)}"
            )
        if not required <= set(ids):
            raise ValueError(
                f"frame {frame.frame_idx} misses visibly confirmed IDs: "
                f"required={sorted(required)}, returned={ids}"
            )


def localize(client, model, video, plan, info, work):
    indices = list(range(0, info["num_frames"], 30))
    directory = work / "sparse_frames"
    directory.mkdir()
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-i", str(video), "-map", "0:v:0", "-an", "-threads", "1",
        # Assign successive timestamps after selection so the image muxer emits each frame once.
        "-filter_threads", "1", "-vf", r"select=not(mod(n\,30)),settb=1/1,setpts=N", "-r", "1",
        "-q:v", "2", "-start_number", "0", str(directory / "selected_%06d.jpg"),
    ], check=True)
    images = sorted(directory.glob("selected_*.jpg"))
    if len(images) != len(indices):
        raise ValueError("Decoded sparse frame count differs from the video metadata")
    for index, image in zip(indices, images):
        image.rename(directory / f"frame_{index:06d}.jpg")
    config = types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_level="high"),
        response_mime_type="application/json", response_schema=SparseResponse,
    )
    tracks = {}
    for start in range(0, len(indices), 8):
        frames = indices[start:start + 8]
        prompt = build_prompt(plan, frames, info)
        response = client.models.generate_content(
            model=model, contents=build_contents(types, prompt, frames, directory),
            config=config,
        )
        parsed = (SparseResponse.model_validate(response.parsed) if response.parsed is not None
                  else SparseResponse.model_validate_json(response.text))
        validate_response(parsed, plan, frames)
        for frame in parsed.frames:
            for obj in frame.objects:
                tracks.setdefault(obj.object_id, {})[frame.frame_idx] = obj.box_2d
    return tracks
