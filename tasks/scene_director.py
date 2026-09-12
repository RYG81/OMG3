"""Structured output contract for the existing Scene Director node."""

from __future__ import annotations

from typing import Any

SCENE_DIRECTOR_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "scene_prompt": {"type": "string", "minLength": 1},
        "composition_guide": {"type": "string"},
        "lighting_setup": {"type": "string"},
        "character_descriptions": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "director_notes": {"type": "string"},
    },
    "required": [
        "scene_prompt",
        "composition_guide",
        "lighting_setup",
        "character_descriptions",
        "negative_prompt",
        "director_notes",
    ],
    "additionalProperties": False,
}
