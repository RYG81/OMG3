"""Structured output contract for the existing Prompt Critic node."""

from __future__ import annotations

from typing import Any

PROMPT_CRITIC_FIELDS = (
    "overall_score",
    "clarity_feedback",
    "detail_feedback",
    "consistency_feedback",
    "improvement_suggestions",
    "improved_prompt",
    "missing_elements",
    "redundant_elements",
)

PROMPT_CRITIC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {field: {"type": "string"} for field in PROMPT_CRITIC_FIELDS},
    "required": list(PROMPT_CRITIC_FIELDS),
    "additionalProperties": False,
}
