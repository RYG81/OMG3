"""Structured contracts for prompt combination, translation, wildcards, aspect, and LoRA advice."""

from __future__ import annotations

from typing import Any

COMBINER_FIELDS = (
    "combined_prompt",
    "prompt_analysis",
    "conflict_resolution",
    "element_breakdown",
    "negative_prompt",
)
TRANSLATOR_FIELDS = (
    "translated_prompt",
    "source_analysis",
    "translation_notes",
    "format_additions",
    "potential_issues",
    "negative_prompt",
)
WILDCARD_FIELDS = ("wildcard_prompt", "static_elements", "varied_elements", "total_combinations")
ASPECT_FIELDS = ("optimized_prompt", "composition_notes", "cropping_suggestions", "negative_additions")
LORA_FIELDS = (
    "prompt_analysis",
    "style_loras",
    "concept_loras",
    "quality_loras",
    "priority_loras",
    "lora_keywords",
    "weight_suggestions",
    "prompt_optimization",
)


def _schema(fields: tuple[str, ...], primary: str, extras: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            field: {"type": "string", **({"minLength": 1} if field == primary else {})}
            for field in (*fields, *extras)
        },
        "required": list(fields),
        "additionalProperties": False,
    }


COMBINER_SCHEMA = _schema(COMBINER_FIELDS, "combined_prompt", ("confidence_score",))
TRANSLATOR_SCHEMA = _schema(TRANSLATOR_FIELDS, "translated_prompt")
ASPECT_SCHEMA = _schema(
    ASPECT_FIELDS,
    "optimized_prompt",
    ("removed_elements", "added_elements", "focal_point", "background_notes"),
)

WILDCARD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        **{field: {"type": "string"} for field in WILDCARD_FIELDS},
        "sample_outputs": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "variation_notes": {"type": "string"},
    },
    "required": [*WILDCARD_FIELDS, "sample_outputs"],
    "additionalProperties": False,
}

_LORA_LIST_OR_STRING = {
    "oneOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}, "maxItems": 30},
    ]
}
LORA_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "prompt_analysis": {"type": "string"},
        "style_loras": _LORA_LIST_OR_STRING,
        "concept_loras": _LORA_LIST_OR_STRING,
        "quality_loras": _LORA_LIST_OR_STRING,
        "priority_loras": {"type": "string"},
        "lora_keywords": {"type": "string"},
        "weight_suggestions": {"type": "string"},
        "prompt_optimization": {"type": "string"},
        "character_notes": {"type": "string"},
        "trigger_words_needed": {"type": "string"},
        "alternative_approach": {"type": "string"},
    },
    "required": list(LORA_FIELDS),
    "additionalProperties": False,
}


def list_or_string(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")
