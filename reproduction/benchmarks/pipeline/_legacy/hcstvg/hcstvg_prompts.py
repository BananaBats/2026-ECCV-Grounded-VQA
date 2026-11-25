"""HC-STVG captions refer to one person and an ordered sequence of actions."""
import os

PROMPT_VERSION = 'hcstvg_person_actions_amodal_v1'

PERSON_RULE = '''The caption refers to exactly ONE person. Identify the person whose
appearance and complete ordered action sequence match the caption. Other people,
animals, furniture, and held objects are context, not additional targets. Use
obj_id=0 throughout. Never substitute a face, hand, body part, or a group for the
person. Follow the person across pose changes and partial occlusion.'''


def box_mode():
    mode = os.environ.get('HCSTVG_BOX_MODE', 'amodal')
    if mode not in ('amodal', 'visible'):
        raise ValueError('HCSTVG_BOX_MODE must be amodal or visible')
    return mode


def noun_prompt(caption):
    return f'''Inspect the complete labeled video for this HC-STVG caption:
{caption}

{PERSON_RULE}
Return one concise answer_noun_phrase identifying that person using stable visual
attributes needed to distinguish similar people. In reasoning, check the caption's
actions in order; clothing color alone is insufficient. Do not return a frame or
bbox in this step. Call submit_answer_noun_phrase exactly once.'''


def critic_prompt(caption, candidate):
    return f'''Independently verify the proposed person for this HC-STVG caption.
Caption: {caption}
Candidate person: {candidate}

{PERSON_RULE}
Re-derive the referent from the complete video. Check the person's appearance,
the ordered actions, and interactions, distinguishing the actor from the person
acted upon. Partial occlusion is not evidence that this is a different person.
Return expected_instance_count=1. If correct, verdict="correct". If wrong,
verdict="wrong" and ALWAYS supply a nonempty corrected_noun_phrase identifying
the correct person. Give supporting orig_frame labels, not proxy frame numbers.
Call submit_answer_verification exactly once.'''


def selection_prompt(caption, person, allowed_frames, max_anchors, num_frames):
    return f'''Select the person track, action interval, and initialization frames
for this HC-STVG caption: {caption}
Verified person: {person}

{PERSON_RULE}
Return exactly one target with obj_id=0 and every anchor's obj_ids=[0].
The input contains original frames 0..{num_frames-1}; use orig_frame=N labels.
Every anchor MUST be selected from these allowed original frame indices:
{list(allowed_frames)}

Prefer ONE clear frame showing as much of the whole person as possible, with
minimal occlusion and good separation from other people. Use at most {max_anchors}
anchors when reappearance or strong pose changes require them. Prefer a clear
frame inside the action interval; a clearer frame outside it is allowed if the
same identity is established. A partially visible face alone is a poor anchor
when a clearer full-person view exists elsewhere.

Return inclusive original-frame indices for:
- first_visible_frame_idx / last_visible_frame_idx and visibility_segments:
  all intervals with any directly visible part of this person. Partial occlusion
  does not end visibility; a fully hidden interval may separate visible segments.
- grounding_segment: ONE contiguous interval covering the complete ordered action
  sequence in the caption, from onset of its first described action through
  completion of its last described action. Include intermediate transitions that
  belong to that sequence. Do not select only one short matching sub-action, and
  do not widen the interval to the person's entire appearance just because their
  clothing continues to match. Attribute/reference clauses can identify the
  person without defining the action boundaries. Brief occlusion need not end an
  ongoing action sequence when surrounding evidence establishes continuity.

Do not force any fixed duration or assume the entire 20-second clip is positive.
Call submit_multi_anchor_selection exactly once.'''


def boxes_prompt(caption, person, anchor, identity_text):
    if box_mode() == 'amodal':
        extent = '''Return the person's COMPLETE inferred body extent within the
image, including parts hidden by a desk, furniture, another person, or other
occluders. For example, a desk covering the legs does not make the target just a
head-and-torso box. Infer the hidden person's own position, scale, and pose from
visible anatomy and the same person's before/after frames in the supplied video.
Do not replace the person with the desk's box or expand to a person-plus-occluder
group. Extend only over hidden body regions supported by this evidence; do not
invent extreme limb positions. Clip the inferred box to the image boundaries.'''
    else:
        extent = '''Return a tight box around the person's VISIBLE extent only.
Do not extend over hidden body parts. Keep this visible-only comparison separate
from the amodal setting.'''
    return f'''HC-STVG box mode: {box_mode()}
Caption: {caption}
Verified person: {person}
This exact original image is frame {anchor.frame_idx}.
Requested identity:
{identity_text}

{PERSON_RULE}
{extent}
Return exactly obj_ids={anchor.obj_ids}, each with box_2d=[ymin,xmin,ymax,xmax]
normalized to 0..1000. Include worn clothing; do not separately target carried
objects, other people, furniture, or shadows. Lower confidence when the hidden
extent is uncertain. Call submit_multi_anchor_boxes exactly once.'''
