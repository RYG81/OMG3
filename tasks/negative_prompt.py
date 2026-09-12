"""Structured output contract for the Negative Prompt node."""

from __future__ import annotations

from typing import Any

NEGATIVE_PROMPT_FIELDS = (
    "negative_prompt",
    "quality_negatives",
    "content_negatives",
    "anatomy_negatives",
)

NEGATIVE_PROMPT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        **{field: {"type": "string"} for field in NEGATIVE_PROMPT_FIELDS},
        "reasoning": {"type": "string"},
    },
    "required": list(NEGATIVE_PROMPT_FIELDS),
    "additionalProperties": False,
}
