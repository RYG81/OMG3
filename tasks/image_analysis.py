"""Structured output contract for the existing Image Analyzer node."""

from __future__ import annotations

from typing import Any

_TEXT_FIELDS = (
    "subject",
    "body_details",
    "clothing",
    "pose",
    "location",
    "lighting",
    "camera",
    "color_palette",
    "art_style",
    "mood",
    "technical",
    "reconstruction_prompt",
)

IMAGE_ANALYSIS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        **{field: {"type": "string"} for field in _TEXT_FIELDS},
        "scene_elements": {
            "type": "array",
            "maxItems": 50,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "desc": {"type": "string"},
                },
                "required": ["type", "bbox", "desc"],
                "additionalProperties": False,
            },
        },
    },
    "required": [*_TEXT_FIELDS, "scene_elements"],
    "additionalProperties": False,
}
