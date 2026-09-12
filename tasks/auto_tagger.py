"""Structured output contract for Auto Tagger."""

from __future__ import annotations

from typing import Any

AUTO_TAG_FIELDS = (
    "booru_tags",
    "natural_tags",
    "weighted_tags",
    "character_tags",
    "style_tags",
    "quality_tags",
    "meta_tags",
)

AUTO_TAG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        **{field: {"type": "string"} for field in AUTO_TAG_FIELDS},
        "suggested_negatives": {"type": "string"},
        "tag_count": {"type": ["string", "integer"]},
    },
    "required": list(AUTO_TAG_FIELDS),
    "additionalProperties": False,
}
