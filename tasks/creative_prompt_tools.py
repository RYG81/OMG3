"""Structured contracts for merge, edit, story, pose, and outfit prompt nodes."""

from __future__ import annotations

from typing import Any

IMAGE_MERGER_FIELDS = (
    "merged_prompt",
    "image_1_analysis",
    "image_2_analysis",
    "negative_prompt",
    "composition_notes",
)
IMAGE_EDIT_FIELDS = ("edit_prompt", "preservation_notes", "change_summary")
IMAGE_STORY_FIELDS = (
    "story",
    "title",
    "setting_description",
    "character_profiles",
    "mood_analysis",
    "dialogue_sample",
    "before_scene",
    "after_scene",
    "themes",
)
POSE_FIELDS = (
    "pose_description",
    "body_position",
    "limb_positions",
    "hand_description",
    "face_expression",
    "pose_tags",
)
OUTFIT_FIELDS = (
    "outfit_prompt",
    "top_description",
    "bottom_description",
    "accessories",
    "footwear",
    "outfit_tags",
)


def _schema(fields: tuple[str, ...], primary: str, extras: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            field: {"type": "string", **({"minLength": 1} if field == primary else {})}
            for field in (*fields, *extras)
        },
        "required": list(fields),
        "additionalProperties": False,
    }


IMAGE_MERGER_SCHEMA = _schema(IMAGE_MERGER_FIELDS, "merged_prompt", ("user_intent",))
IMAGE_EDIT_SCHEMA = _schema(IMAGE_EDIT_FIELDS, "edit_prompt")
IMAGE_STORY_SCHEMA = _schema(IMAGE_STORY_FIELDS, "story", ("narrative_hooks",))
POSE_SCHEMA = _schema(POSE_FIELDS, "pose_description", ("camera_angle_suggestion", "pose_difficulty"))
OUTFIT_SCHEMA = _schema(OUTFIT_FIELDS, "outfit_prompt", ("style_notes", "color_palette"))
