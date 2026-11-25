"""Pure support for fresh Gemini-only boxes at every decoded source frame.

The first chunk infers targets afresh from the question and images, without old
A answers, a C plan, SAM, or GT. Later chunks keep that target registry fixed.
Every source frame is requested without resampling; whole-video sparse context
and a short prediction handoff accompany bounded dense chunks. This chunked
protocol changes context and identity handoff as well as the old output rate,
and must be reported as such when comparing against the one-call 1fps baseline.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


METHOD_REVISION = 'naive-amodal-all-source-frames-chunked-v1'
PROMPT_OVERHEAD_BYTES = 65536


class NaiveTarget(BaseModel):
    model_config = ConfigDict(extra='forbid')
    object_id: int = Field(ge=0, le=9)
    label: str
    description: str


class NaiveObject(BaseModel):
    model_config = ConfigDict(extra='forbid')
    object_id: int = Field(ge=0, le=9)
    box_2d: list[int] = Field(min_length=4, max_length=4)

    @field_validator('box_2d')
    @classmethod
    def valid_box(cls, value: list[int]) -> list[int]:
        values = [int(v) for v in value]
        if any(v < 0 or v > 1000 for v in values):
            raise ValueError('box_2d values must be in 0..1000')
        y0, x0, y1, x1 = values
        if y1 <= y0 or x1 <= x0:
            raise ValueError('box_2d must have positive area')
        return values


class NaiveFrame(BaseModel):
    model_config = ConfigDict(extra='forbid')
    frame_idx: int = Field(ge=0)
    objects: list[NaiveObject] = Field(max_length=10)


class NaiveResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: str
    targets: list[NaiveTarget] = Field(min_length=1, max_length=10)
    frames: list[NaiveFrame] = Field(min_length=1)


def _indices(values: list[int], name: str) -> list[int]:
    result = list(values)
    if any(type(v) is not int or v < 0 for v in result):
        raise ValueError(f'{name} must contain nonnegative integer frame indices')
    if result != sorted(set(result)):
        raise ValueError(f'{name} must be unique and strictly increasing')
    return result


def _registry_targets(registry: dict[str, Any]) -> list[NaiveTarget]:
    if not isinstance(registry, dict) or not isinstance(registry.get('targets'), list):
        raise ValueError('registry must contain a targets list')
    targets = [NaiveTarget.model_validate(t) for t in registry['targets']]
    ids = [target.object_id for target in targets]
    if not 1 <= len(targets) <= 10 or len(ids) != len(set(ids)):
        raise ValueError('registry must contain 1..10 unique target IDs')
    return targets


def make_prompt(
    question: str, requested: list[int], context_indices: list[int],
    video: dict[str, Any], registry: dict[str, Any] | None,
    handoff: list[dict[str, Any]],
) -> str:
    """Describe independent whole-video context and exact output frame roles."""
    requested = _indices(requested, 'requested')
    context_indices = _indices(context_indices, 'context_indices')
    if not requested:
        raise ValueError('requested must not be empty')
    requested_text = ', '.join(map(str, requested))
    context_text = ', '.join(map(str, context_indices)) or '(none)'
    if registry is None:
        identity = (
            'This is the first chunk. Decide afresh which physical object or objects '
            'answer the question using the whole-video context and current images. '
            'Assign one object_id (0, 1, 2, ...) to each distinct target. Return a short '
            'answer and describe the distinguishing visual attributes of every target '
            'in targets. This fresh target registry will be fixed for later chunks.'
        )
    else:
        targets = _registry_targets(registry)
        fixed = {'targets': [target.model_dump(mode='json') for target in targets]}
        if 'answer' in registry:
            fixed['answer'] = registry['answer']
        identity = (
            'The first chunk already established the following immutable target '
            'registry from this same video and question. Track exactly these physical '
            'identities. Copy all targets records verbatim, retaining their object_id, '
            'label, and description. Never add, remove, renumber, merge, or swap targets. '
            'Use the same answer when provided.\n'
            + json.dumps(fixed, ensure_ascii=False, separators=(',', ':'))
        )
    handoff_text = json.dumps(handoff, ensure_ascii=False, separators=(',', ':'))
    return f"""You are given still frames from one video with exact source frame indices.
The source resolution is {video.get('width', '?')}x{video.get('height', '?')} and the
source frame rate is {video.get('fps', '?')} fps. Source frames were not resampled.

WHOLE-VIDEO CONTEXT: frames {context_text} are a sparse overview of the full video,
normally every 30th source frame (about 1fps). Use them to understand the question
and target identities. These context images do not define the output schedule.

REQUESTED OUTPUT: return records for ALL source frame indices {requested_text},
in exactly that order. These are the consecutive dense frames of the current
chunk, approximately 30fps. Do not skip or interpolate requested frame records.
If an image serves multiple roles, its frame_idx identifies the same source image.

Question: {question}

Task: decide which physical object or objects answer the question, then draw an
AMODAL box around each target's complete projected extent in every requested
frame where the target is present and its extent is supported by the video.

{identity}

Rules:
- Answer with the objects the question asks for, not the person handling them and not
  look-alike distractors. If the question asks for several objects, list each one.
- Give each distinct physical object one object_id and keep that same object_id for
  the same object across all frames. Do not reuse a target ID for another object.
- Return exactly one frames record per requested frame_idx, in the listed order.
  Include a target when it is present in the camera view, including partial
  occlusion. Recover the portions hidden by a hand, another object, or an occluder
  using the target's shape, its surrounding frames, and whole-video context.
  During full occlusion, include the target only when its current position and
  extent are supported by nearby motion and temporal context. Omit a target that
  has truly left the view or whose current extent cannot be supported; do not
  invent a location. If no target has a supported box, return objects: [].
- box_2d is [ymin, xmin, ymax, xmax] with integer coordinates normalized to 0..1000,
  a tight box around the COMPLETE projected object, including its occluded parts,
  rather than only its visible fragment. Infer each hidden left, top, right, and
  bottom boundary from this same physical object's appearance and local temporal
  evidence. Do not enlarge the box beyond that object's supported extent. Clip
  the projected extent to the image boundaries. One box covers exactly one target;
  never box a group, a pile, or the target together with a hand or another item.
  Never substitute an occluding distractor for a target; an occluder can itself
  be selected if the question requests it.
  Do not use an occluder's outline as the target's boundary. An
  occluder may overlap the target's amodal box without changing its full extent.
- Context and previous handoff frames must not appear as extra output records.
  For requested frames, inspect their images and predict their boxes directly.

PREVIOUS CHUNK HANDOFF: the following are this model's own earlier predictions,
provided only to maintain identity across chunk boundaries. They are not GT and
must not be copied as predictions for different source frames:
{handoff_text}
""".strip()


def validate_response(
    response: NaiveResponse | dict[str, Any], requested: list[int],
    registry: dict[str, Any] | None = None,
) -> None:
    """Require every requested frame, unambiguous IDs, and a fixed later registry."""
    response = NaiveResponse.model_validate(response)
    requested = _indices(requested, 'requested')
    returned = [frame.frame_idx for frame in response.frames]
    if returned != requested:
        raise ValueError(f'frame sequence mismatch: expected={requested}, returned={returned}')
    ids = [target.object_id for target in response.targets]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate target IDs')
    known = set(ids)
    if registry is not None:
        fixed = _registry_targets(registry)
        canonical = lambda ts: sorted((t.object_id, t.label, t.description) for t in ts)
        if canonical(response.targets) != canonical(fixed):
            raise ValueError('response targets differ from the immutable target registry')
        known = {target.object_id for target in fixed}
    for frame in response.frames:
        frame_ids = [obj.object_id for obj in frame.objects]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f'frame {frame.frame_idx} has duplicate object IDs')
        unknown = set(frame_ids) - known
        if unknown:
            raise ValueError(f'frame {frame.frame_idx} has unknown object IDs: {sorted(unknown)}')


def partition_frames(
    frame_indices: list[int], sizes: dict[int, int], context_indices: list[int], *,
    max_frames: int = 48, max_inline_bytes: int = 18 * 1024 * 1024,
    handoff_frames: int = 3,
) -> list[list[int]]:
    """Greedily bound requested count and base64 bytes of all unique images.

    Each request includes the whole-video context, up to handoff_frames preceding
    source images, and its own requested frames. 65536 bytes are reserved for the
    prompt, JSON wrapper, frame labels, and schema. Sizes are raw JPEG bytes.
    """
    frames = _indices(frame_indices, 'frame_indices')
    context = set(_indices(context_indices, 'context_indices'))
    for name, value, minimum in [
        ('max_frames', max_frames, 1), ('max_inline_bytes', max_inline_bytes, 1),
        ('handoff_frames', handoff_frames, 0),
    ]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not frames:
        return []
    encoded = {}
    for frame in set(frames) | context:
        size = sizes.get(frame)
        if type(size) is not int or size < 0:
            raise ValueError(f'missing or invalid JPEG byte size for frame {frame}')
        encoded[frame] = 4 * ((size + 2) // 3)
    chunks = []
    start = 0
    while start < len(frames):
        included = context | set(frames[max(0, start - handoff_frames):start])
        used = PROMPT_OVERHEAD_BYTES + sum(encoded[frame] for frame in included)
        chunk = []
        for frame in frames[start:start + max_frames]:
            extra = 0 if frame in included else encoded[frame]
            if used + extra > max_inline_bytes:
                break
            included.add(frame)
            used += extra
            chunk.append(frame)
        if not chunk:
            frame = frames[start]
            required = used + (0 if frame in included else encoded[frame])
            raise ValueError(
                f'frame {frame} cannot fit with context and handoff: '
                f'requires {required} bytes, budget {max_inline_bytes}'
            )
        chunks.append(chunk)
        start += len(chunk)
    return chunks


def tracks_from_frames(
    frame_records: list[NaiveFrame | dict[str, Any]], interpolate: bool = False,
) -> list[dict[str, Any]]:
    """Export xyxy 0..1 tracks; direct mode preserves explicitly absent frames.

    interpolate=True mirrors the old naive export: linearly fill between any
    successive observations of an object, even across frames where it was omitted.
    """
    frames = [NaiveFrame.model_validate(frame) for frame in frame_records]
    frame_ids = [frame.frame_idx for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError('duplicate source frame records')
    per_object: dict[int, list[tuple[int, list[float]]]] = {}
    for frame in frames:
        ids = [obj.object_id for obj in frame.objects]
        if len(ids) != len(set(ids)):
            raise ValueError(f'frame {frame.frame_idx} has duplicate object IDs')
        for obj in frame.objects:
            y0, x0, y1, x1 = obj.box_2d
            box = [x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0]
            per_object.setdefault(obj.object_id, []).append((frame.frame_idx, box))
    tracks = []
    for object_id, observations in sorted(per_object.items()):
        observations.sort()
        output: dict[int, list[float]] = {}
        for index, (frame, box) in enumerate(observations):
            if interpolate and index + 1 < len(observations):
                following, next_box = observations[index + 1]
                for current in range(frame, following):
                    weight = (current - frame) / (following - frame)
                    output[current] = [round(a + (b - a) * weight, 6)
                                       for a, b in zip(box, next_box)]
            else:
                output[frame] = [round(value, 6) for value in box]
        indices = sorted(output)
        tracks.append({'id': object_id, 'frame_ids': indices,
                       'bounding_boxes': [output[frame] for frame in indices]})
    return tracks
