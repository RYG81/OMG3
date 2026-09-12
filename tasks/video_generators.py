"""Structured contracts for LTX Ingredients and model-specific video prompt generators."""

from __future__ import annotations

from typing import Any

LTX_INGREDIENT_FIELDS = ("prompt", "negative_prompt", "recommended_settings")
VIDEO_PROMPT_FIELDS = (
    "positive_prompt",
    "negative_prompt",
    "shot_plan",
    "motion_notes",
    "model_settings",
)


def _schema(fields: tuple[str, ...], primary: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            field: {"type": "string", **({"minLength": 1} if field == primary else {})}
            for field in fields
        },
        "required": list(fields),
        "additionalProperties": False,
    }


LTX_INGREDIENT_SCHEMA = _schema(LTX_INGREDIENT_FIELDS, "prompt")
VIDEO_PROMPT_SCHEMA = _schema(VIDEO_PROMPT_FIELDS, "positive_prompt")
