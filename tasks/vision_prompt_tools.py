"""Structured contracts for image comparison, palette, inpaint, style, and environment nodes."""

from __future__ import annotations

from typing import Any

COMPARATOR_FIELDS = (
    "comparison_summary",
    "similarities",
    "differences",
    "image_a_unique",
    "image_b_unique",
    "quality_comparison",
    "recommendation",
)
COLOR_PALETTE_FIELDS = (
    "palette_hex",
    "palette_names",
    "palette_description",
    "primary_color",
    "accent_colors",
    "mood_description",
    "color_prompt_tags",
)
INPAINT_FIELDS = ("inpaint_prompt", "context_description", "negative_prompt")
STYLE_TRANSFER_FIELDS = (
    "styled_prompt",
    "style_description",
    "style_tags",
    "color_palette",
    "technique_notes",
)
ENVIRONMENT_FIELDS = (
    "transformed_prompt",
    "original_analysis",
    "lighting_description",
    "atmosphere_description",
    "negative_prompt",
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


IMAGE_COMPARATOR_SCHEMA = _schema(COMPARATOR_FIELDS, "comparison_summary", ("similarity_score",))
COLOR_PALETTE_SCHEMA = _schema(
    COLOR_PALETTE_FIELDS,
    "palette_hex",
    ("color_harmony", "temperature", "saturation_level", "usage_suggestions"),
)
INPAINT_SCHEMA = _schema(INPAINT_FIELDS, "inpaint_prompt", ("style_tags", "blending_notes"))
STYLE_TRANSFER_SCHEMA = _schema(STYLE_TRANSFER_FIELDS, "styled_prompt")
ENVIRONMENT_SCHEMA = _schema(
    ENVIRONMENT_FIELDS,
    "transformed_prompt",
    ("color_grading", "affected_elements"),
)
