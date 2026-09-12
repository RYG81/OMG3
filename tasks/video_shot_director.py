"""Structured output contract for the existing Video Shot Director node."""

from __future__ import annotations

from typing import Any

VIDEO_SHOT_FIELDS = (
    "sequence_prompt",
    "shot_list",
    "per_shot_prompts",
    "continuity_notes",
    "negative_prompt",
    "model_notes",
)

VIDEO_SHOT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        field: {"type": "string", **({"minLength": 1} if field == "sequence_prompt" else {})}
        for field in VIDEO_SHOT_FIELDS
    },
    "required": list(VIDEO_SHOT_FIELDS),
    "additionalProperties": False,
}
