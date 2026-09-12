"""Structured output contract builder for Prompt Variations."""

from __future__ import annotations

from typing import Any

VARIATION_FIELDS = tuple(f"variation_{index}" for index in range(1, 11))


def build_prompt_variations_schema(num_variations: int) -> dict[str, Any]:
    count = max(2, min(10, int(num_variations)))
    properties: dict[str, Any] = {
        "original_analysis": {"type": "string"},
        "variation_strategy": {"type": "string"},
        "all_variations": {"type": "string"},
    }
    properties.update({field: {"type": "string"} for field in VARIATION_FIELDS})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": [f"variation_{index}" for index in range(1, count + 1)],
        "additionalProperties": False,
    }
