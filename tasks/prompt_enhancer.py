"""Structured output contract for the existing Prompt Enhancer node."""

from __future__ import annotations

from typing import Any

PROMPT_ENHANCER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "enhanced_prompt": {"type": "string", "minLength": 1},
        "negative_prompt": {"type": "string"},
        "tags": {"type": "string"},
        "changes_summary": {"type": "string"},
    },
    "required": ["enhanced_prompt", "negative_prompt", "tags", "changes_summary"],
    "additionalProperties": False,
}
