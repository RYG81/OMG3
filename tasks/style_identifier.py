"""Structured output contract for Style Identifier."""

from __future__ import annotations

from typing import Any

STYLE_OUTPUT_FIELDS = (
    "primary_style",
    "style_movement",
    "techniques",
    "possible_influences",
    "medium",
    "style_tags",
    "similar_artists",
    "replication_prompt",
)
_STYLE_EXTRA_FIELDS = ("era_period", "technical_notes", "unique_elements")

STYLE_IDENTIFIER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        field: {"type": "string"} for field in (*STYLE_OUTPUT_FIELDS, *_STYLE_EXTRA_FIELDS)
    },
    "required": list(STYLE_OUTPUT_FIELDS),
    "additionalProperties": False,
}
