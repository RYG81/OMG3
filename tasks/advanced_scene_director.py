"""Structured output contract for Advanced Scene Director."""

from __future__ import annotations

from typing import Any

ADVANCED_SCENE_FIELDS = (
    "scene_prompt",
    "character_breakdown",
    "spatial_layout",
    "lighting_description",
    "camera_technical",
    "atmosphere_details",
    "composition_notes",
    "color_script",
    "negative_prompt",
    "director_vision",
)

ADVANCED_SCENE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        field: {"type": "string", **({"minLength": 1} if field == "scene_prompt" else {})}
        for field in ADVANCED_SCENE_FIELDS
    },
    "required": list(ADVANCED_SCENE_FIELDS),
    "additionalProperties": False,
}
