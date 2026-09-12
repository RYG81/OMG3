"""Dynamic structured output contract for Storyboard Generator."""

from __future__ import annotations

from typing import Any

PANEL_FIELDS = tuple(f"panel_{index}" for index in range(1, 13))


def build_storyboard_schema(num_panels: int) -> dict[str, Any]:
    count = max(2, min(12, int(num_panels)))
    properties: dict[str, Any] = {field: {"type": "string"} for field in PANEL_FIELDS}
    properties.update(
        {
            "all_panels": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                ]
            },
            "character_reference": {"type": "string"},
            "story_summary": {"type": "string"},
            "camera_directions": {"type": "string"},
            "pacing_notes": {"type": "string"},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": [
            *(f"panel_{index}" for index in range(1, count + 1)),
            "character_reference",
            "story_summary",
            "camera_directions",
        ],
        "additionalProperties": False,
    }
