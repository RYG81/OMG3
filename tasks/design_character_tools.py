"""Structured contracts for subject, creature, hands, lighting, and material design nodes."""

from __future__ import annotations

from typing import Any

SUBJECT_FIELDS = (
    "full_description",
    "appearance_tags",
    "physical_summary",
    "face_description",
    "body_description",
    "clothing_suggestion",
    "consistency_anchors",
    "negative_prompt",
)
CREATURE_FIELDS = (
    "creature_prompt",
    "anatomy_description",
    "surface_details",
    "feature_details",
    "coloration",
    "eye_description",
    "movement_impression",
    "size_reference",
    "style_tags",
    "negative_prompt",
)
HAND_FIELDS = (
    "hand_prompt",
    "right_hand",
    "left_hand",
    "finger_positions",
    "palm_orientation",
    "wrist_angle",
    "hand_interaction",
    "simplification_tips",
    "hand_tags",
    "negative_prompt",
    "alternative_pose",
)
LIGHTING_FIELDS = (
    "lighting_prompt",
    "technical_setup",
    "color_palette",
    "shadow_description",
    "atmosphere_effects",
    "mood_contribution",
    "style_tags",
    "avoid",
)
MATERIAL_FIELDS = (
    "material_prompt",
    "surface_texture",
    "color_description",
    "reflectivity",
    "transparency",
    "pattern_details",
    "wear_details",
    "tactile_impression",
    "close_up_details",
    "material_tags",
    "negative_prompt",
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


SUBJECT_SCHEMA = _schema(SUBJECT_FIELDS, "full_description", ("personality_expression",))
CREATURE_SCHEMA = _schema(CREATURE_FIELDS, "creature_prompt", ("environmental_context",))
HAND_SCHEMA = _schema(HAND_FIELDS, "hand_prompt", ("difficulty_rating",))
LIGHTING_SCHEMA = _schema(LIGHTING_FIELDS, "lighting_prompt", ("time_of_day_match",))
MATERIAL_SCHEMA = _schema(MATERIAL_FIELDS, "material_prompt", ("lighting_tips",))
