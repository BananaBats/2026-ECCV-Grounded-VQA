"""Question-guided Gemini planning and exact-frame anchor localization."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .media import labeled_proxy, read_frames


class NounPhraseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_noun_phrase: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("answer_noun_phrase")
    @classmethod
    def validate_noun_phrase(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer_noun_phrase must not be empty")
        return value


def call_with_retry(
    label: str, callback: Callable[[], Any], max_attempts: int = 8
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[Gemini] {label} attempt={attempt}/{max_attempts}", flush=True)
            return callback()
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            transient = any(
                token in lowered
                for token in (
                    "408", "429", "500", "502", "503", "504", "timeout",
                    "temporarily unavailable", "too_many_requests",
                )
            )
            if not transient or attempt == max_attempts:
                raise
            match = re.search(r"(?:retry|retrying)\s+in\s+([0-9.]+)s", message, re.I)
            delay = float(match.group(1)) + 2 if match else min(60, 5 * 2 ** (attempt - 1))
            print(f"[Gemini] retrying in {delay:.1f}s: {exc}", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable Gemini retry state")


def function_config(name: str, schema: dict[str, Any]) -> types.GenerateContentConfig:
    declaration = types.FunctionDeclaration(
        name=name,
        description=f"Return the validated {name} result.",
        parameters_json_schema=schema,
    )
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=[name]
            )
        ),
        temperature=0.0,
    )


def call_function(
    client: genai.Client,
    model: str,
    name: str,
    schema: dict[str, Any],
    parts: list[types.Part],
) -> dict[str, Any]:
    response = call_with_retry(
        name,
        lambda: client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=function_config(name, schema),
        ),
    )
    calls = response.function_calls or []
    if len(calls) != 1 or calls[0].name != name:
        returned = [call.name for call in calls]
        raise RuntimeError(f"expected one {name} call, got {returned}")
    return dict(calls[0].args or {})


class VisibilitySegment(BaseModel):
    """One contiguous appearance interval of a target in the original video."""

    model_config = ConfigDict(extra="forbid")

    first_frame_idx: int = Field(ge=0)
    last_frame_idx: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> "VisibilitySegment":
        if self.first_frame_idx > self.last_frame_idx:
            raise ValueError("segment first_frame_idx must not exceed last_frame_idx")
        return self


class TargetIdentity(BaseModel):
    """A target identity shared across every anchor frame."""

    model_config = ConfigDict(extra="forbid")

    obj_id: int = Field(ge=0)
    object_label: str
    visual_identity_description: str
    visibility_segments: list[VisibilitySegment] = Field(default_factory=list)

    @field_validator("object_label", "visual_identity_description")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target identity fields must not be empty")
        return value

    @field_validator("visibility_segments")
    @classmethod
    def sort_segments(cls, value: list[VisibilitySegment]) -> list[VisibilitySegment]:
        return sorted(value, key=lambda item: item.first_frame_idx)


class AnchorSelection(BaseModel):
    """A source frame and the identities to initialize there."""

    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    obj_ids: list[int] = Field(min_length=1)
    reason: str

    @field_validator("obj_ids")
    @classmethod
    def unique_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("anchor obj_ids must be unique")
        return [int(item) for item in value]


class AnchorSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_summary: str
    first_visible_frame_idx: int = Field(ge=0)
    last_visible_frame_idx: int = Field(ge=0)
    visibility_boundary_reason: str
    targets: list[TargetIdentity] = Field(min_length=1)
    anchors: list[AnchorSelection] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_global_identity_and_coverage(self) -> "AnchorSelectionResponse":
        target_ids = [target.obj_id for target in self.targets]
        if sorted(target_ids) != list(range(len(target_ids))):
            raise ValueError("targets must use contiguous obj_ids starting at 0")
        if self.first_visible_frame_idx > self.last_visible_frame_idx:
            raise ValueError("first_visible_frame_idx must not exceed last_visible_frame_idx")
        anchor_frames = [anchor.frame_idx for anchor in self.anchors]
        if len(anchor_frames) != len(set(anchor_frames)):
            raise ValueError("anchor frame_idx values must be unique")
        known_ids = set(target_ids)
        covered_ids = {obj_id for anchor in self.anchors for obj_id in anchor.obj_ids}
        if not covered_ids <= known_ids:
            raise ValueError("anchors refer to an unknown obj_id")
        if covered_ids != known_ids:
            raise ValueError("every target obj_id must be covered by at least one anchor")
        return self


class AnchorBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obj_id: int = Field(ge=0)
    box_2d: list[int]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("box_2d")
    @classmethod
    def valid_box(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError("box_2d must be [ymin, xmin, ymax, xmax]")
        if any(not 0 <= int(item) <= 1000 for item in value):
            raise ValueError("box_2d values must be in 0..1000")
        ymin, xmin, ymax, xmax = map(int, value)
        if ymax <= ymin or xmax <= xmin:
            raise ValueError("box_2d must have positive area")
        return [ymin, xmin, ymax, xmax]


class AnchorBoxesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boxes: list[AnchorBox] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_obj_ids(self) -> "AnchorBoxesResponse":
        ids = [box.obj_id for box in self.boxes]
        if len(ids) != len(set(ids)):
            raise ValueError("a frame can contain at most one bbox per obj_id")
        return self


NOUN_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "wrong"]},
        "corrected_noun_phrase": {"type": "string"},
        "evidence_frames": {"type": "array", "items": {"type": "integer"}},
        "expected_instance_count": {"type": "integer", "minimum": 1},
    },
    "required": ["reasoning", "verdict", "expected_instance_count"],
}


def proxy_video_part(proxy_path: Path, fps: float) -> types.Part:
    """Video part with an explicit Gemini sampling rate (default API rate is 1fps)."""
    part = types.Part.from_bytes(data=proxy_path.read_bytes(), mime_type="video/mp4")
    part.video_metadata = types.VideoMetadata(fps=fps)
    return part


TARGET_RULES = {
    "query": (
        "Track the entity or entities the query refers to. A person, a body part, "
        "or an object can be the target. Use the action and its participants to "
        "distinguish the requested entity from nearby people and objects."
    ),
    "object": (
        "Track the requested object or objects. A person or hand manipulating "
        "an object supplies context and is outside this object-only target set."
    ),
    "person": (
        "Track each person described by the query. Use appearance, actions "
        "and interactions to distinguish them from other people. Items they "
        "handle supply context. Return each requested person as a whole-person target."
    ),
}


def request_noun_phrase(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    question: str,
    video_fps: float,
    target_kind: str = "query",
) -> NounPhraseResponse:
    prompt = f"""
Inspect the complete labeled planning video and answer this tracking question:
{question}

{TARGET_RULES[target_kind]}
Return one concise answer_noun_phrase identifying the target set. Include the
visible details needed to recognize it. Leave frames and boxes to the later
steps. Call submit_answer_noun_phrase exactly once.
""".strip()
    arguments = call_function(
        client,
        model,
        "submit_answer_noun_phrase",
        NounPhraseResponse.model_json_schema(),
        [
            proxy_video_part(proxy_path, video_fps),
            types.Part.from_text(text=prompt),
        ],
    )
    return NounPhraseResponse.model_validate(arguments)


def verify_noun_phrase(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    question: str,
    candidate: str,
    video_fps: float,
    target_kind: str = "query",
) -> dict[str, Any]:
    """Check the proposed target against the complete planning video."""
    prompt = f"""
A previous analyst proposed this answer target for the tracking question below.
Candidate answer: "{candidate}"
Question: {question}

{TARGET_RULES[target_kind]}
Independently identify the target from the video. Check who acts, what is
acted on, and any temporal or counterfactual condition in the query. Compare
your answer with the candidate. Report the number of distinct target instances
in expected_instance_count.
If the candidate is right, verdict="correct". If not, verdict="wrong" and give
corrected_noun_phrase plus the orig_frame label numbers that support it.
Call submit_answer_verification exactly once.
""".strip()
    arguments = call_function(
        client,
        model,
        "submit_answer_verification",
        NOUN_VERIFICATION_SCHEMA,
        [
            proxy_video_part(proxy_path, video_fps),
            types.Part.from_text(text=prompt),
        ],
    )
    return arguments


def request_anchor_selection(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    question: str,
    noun_phrase: str,
    allowed_frames: Sequence[int],
    max_anchor_frames: int,
    video_fps: float,
    instance_count_hint: int | None = None,
) -> AnchorSelectionResponse:
    hint_text = (
        f"\nThe answer target comprises {instance_count_hint} distinct physical "
        "instances; give each its own obj_id."
        if instance_count_hint and instance_count_hint > 1
        else ""
    )
    prompt = f"""
The complete-video answer target is authoritative: "{noun_phrase}".
Question: {question}

Identify every distinct physical answer instance and give each a stable obj_id
(0, 1, ...).  If the answer denotes multiple separate instances (e.g. several
letters, cups, or shapes), you MUST create one target per instance; never merge
them into a single target.{hint_text}  Then choose anchor frames for SAM3 bbox
prompts.

First try to choose ONE frame in which every target instance is visible,
separated, and minimally occluded.  Only if no such frame exists, choose the
smallest set of up to {max_anchor_frames} frames that covers all target
instances.  In each anchor's obj_ids, include only the target instances visibly
present in that exact frame. Include only members of the verified target set.

Every anchor frame MUST be one of these original-video stride frames:
{list(allowed_frames)}

The proxy frames are labeled orig_frame=N.  Also return inclusive visibility
bounds for the target set over the original video.  For EACH target, return
visibility_segments: every contiguous interval of original-video frames in which
that target is visible, as inclusive first_frame_idx/last_frame_idx pairs.  A
target shown, removed, and shown again has multiple segments.  If a target stays
visible until the video ends, end its final segment at the last original frame
index.  Call submit_multi_anchor_selection exactly once.
""".strip()
    arguments = call_function(
        client,
        model,
        "submit_multi_anchor_selection",
        AnchorSelectionResponse.model_json_schema(),
        [
            proxy_video_part(proxy_path, video_fps),
            types.Part.from_text(text=prompt),
        ],
    )
    selection = AnchorSelectionResponse.model_validate(arguments)
    return align_anchor_selection(selection, allowed_frames, max_anchor_frames)


def align_anchor_selection(
    selection: AnchorSelectionResponse,
    allowed_frames: Sequence[int],
    max_anchor_frames: int,
) -> AnchorSelectionResponse:
    """Snap proxy mistakes to stride frames while preserving target coverage."""
    if len(selection.anchors) > max_anchor_frames:
        raise ValueError(
            f"planner chose {len(selection.anchors)} anchors; max is {max_anchor_frames}"
        )
    allowed = set(map(int, allowed_frames))
    aligned: dict[int, AnchorSelection] = {}
    for anchor in selection.anchors:
        frame_idx = int(anchor.frame_idx)
        if frame_idx not in allowed:
            frame_idx = min(allowed_frames, key=lambda frame: (abs(frame - frame_idx), frame))
        previous = aligned.get(frame_idx)
        if previous is None:
            aligned[frame_idx] = anchor.model_copy(update={"frame_idx": frame_idx})
        else:
            merged_ids = sorted(set(previous.obj_ids) | set(anchor.obj_ids))
            aligned[frame_idx] = previous.model_copy(update={"obj_ids": merged_ids})
    aligned_selection = selection.model_copy(
        update={"anchors": [aligned[index] for index in sorted(aligned)]}
    )
    aligned_selection = AnchorSelectionResponse.model_validate(
        aligned_selection.model_dump(mode="json")
    )
    if len(aligned_selection.anchors) > max_anchor_frames:
        raise ValueError("frame alignment exceeded max_anchor_frames")
    return aligned_selection


def request_anchor_boxes(
    client: genai.Client,
    model: str,
    proxy_path: Path,
    exact_frame_path: Path,
    question: str,
    noun_phrase: str,
    anchor: AnchorSelection,
    identities: Sequence[TargetIdentity],
    video_fps: float,
) -> AnchorBoxesResponse:
    identity_by_id = {item.obj_id: item for item in identities}
    requested = [identity_by_id[obj_id] for obj_id in anchor.obj_ids]
    identity_text = "\n".join(
        f"- obj_id={item.obj_id}: {item.object_label}; {item.visual_identity_description}"
        for item in requested
    )
    prompt = f"""
The answer target is "{noun_phrase}" for this question:
{question}

This exact original image is frame {anchor.frame_idx}.  Draw a tight normalized
box_2d=[ymin,xmin,ymax,xmax] for EVERY and ONLY the requested target identities:
{identity_text}

Return exactly these obj_ids: {anchor.obj_ids}.  Use the supplied stable obj_id
unchanged. Fit each box to that target's visible extent in this image. Exclude
other participants, surrounding occluders, and cast shadows.
Call submit_multi_anchor_boxes exactly once.
""".strip()
    arguments = call_function(
        client,
        model,
        "submit_multi_anchor_boxes",
        AnchorBoxesResponse.model_json_schema(),
        [
            proxy_video_part(proxy_path, video_fps),
            types.Part.from_text(text=f"Exact original frame {anchor.frame_idx}:"),
            types.Part.from_bytes(data=exact_frame_path.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
    )
    result = AnchorBoxesResponse.model_validate(arguments)
    actual_ids = sorted(box.obj_id for box in result.boxes)
    expected_ids = sorted(anchor.obj_ids)
    if actual_ids != expected_ids:
        raise ValueError(
            f"anchor frame {anchor.frame_idx} returned obj_ids={actual_ids}; "
            f"expected {expected_ids}"
        )
    return result


def plan_targets(client, model, video, question, info, work, target_kind="query"):
    proxy = labeled_proxy(video, work / "planning.mp4", info)
    video_fps = 4.0
    answer = request_noun_phrase(client, model, proxy, question, video_fps, target_kind)
    phrase = answer.answer_noun_phrase
    check = verify_noun_phrase(client, model, proxy, question, phrase, video_fps, target_kind)
    correction = check.get("corrected_noun_phrase") or ""
    if check["verdict"] == "wrong" and correction.strip():
        phrase = correction.strip()

    allowed = list(range(0, info["num_frames"], 10))
    count = check.get("expected_instance_count")
    hint = count if isinstance(count, int) and count >= 2 else None
    selection = request_anchor_selection(
        client, model, proxy, question, phrase, allowed, 3, video_fps, hint
    )
    identities = {t.obj_id: t for t in selection.targets}
    anchors = {a.frame_idx: a for a in selection.anchors}
    plan = selection.model_dump(mode="json", exclude={"anchors"})
    plan.update(question=question, answer_noun_phrase=phrase, anchors=[])
    for index, frame in read_frames(video, anchors):
        anchor = anchors[index]
        exact = work / f"anchor_{index}.png"
        if not cv2.imwrite(str(exact), frame):
            raise OSError(f"Could not write the image for anchor {index}")
        boxes = request_anchor_boxes(
            client, model, proxy, exact, question, phrase, anchor,
            [identities[i] for i in anchor.obj_ids], video_fps
        )
        plan["anchors"].append({**anchor.model_dump(), **boxes.model_dump()})
    return plan
