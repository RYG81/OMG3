"""Structured contracts for action, background, emotion, detail, and ControlNet helpers."""

from __future__ import annotations

from typing import Any

ACTION_FIELDS = (
    "action_prompt",
    "pose_description",
    "motion_direction",
    "impact_point",
    "facial_expressions",
    "clothing_dynamics",
    "environmental_interaction",
    "energy_visualization",
    "camera_position",
    "composition_tips",
    "negative_prompt",
)
BACKGROUND_FIELDS = (
    "background_prompt",
    "foreground_elements",
    "midground_elements",
    "background_elements",
    "atmospheric_effects",
    "color_description",
    "depth_cues",
    "integration_tips",
    "negative_prompt",
)
EMOTION_FIELDS = (
    "emotional_prompt",
    "facial_direction",
    "eye_description",
    "body_language",
    "color_mood",
    "lighting_mood",
    "environmental_mood",
    "emotional_tags",
    "avoid",
)
DETAIL_FIELDS = (
    "enhanced_prompt",
    "added_details",
    "detail_tags",
    "quality_boost",
    "negative_additions",
)
CONTROLNET_FIELDS = (
    "optimized_prompt",
    "structural_elements",
    "detail_elements",
    "style_elements",
    "what_to_omit",
    "what_to_emphasize",
    "negative_prompt",
    "common_mistakes",
    "workflow_tips",
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


ACTION_SCHEMA = _schema(ACTION_FIELDS, "action_prompt", ("timing_notes",))
BACKGROUND_SCHEMA = _schema(BACKGROUND_FIELDS, "background_prompt", ("mood_elements",))
EMOTION_SCHEMA = _schema(EMOTION_FIELDS, "emotional_prompt", ("composition_tips",))
DETAIL_SCHEMA = _schema(DETAIL_FIELDS, "enhanced_prompt", ("detail_placement",))
CONTROLNET_SCHEMA = _schema(CONTROLNET_FIELDS, "optimized_prompt", ("strength_notes",))
