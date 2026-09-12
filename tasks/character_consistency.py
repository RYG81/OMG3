"""Structured contracts for character continuity and sheet helper nodes."""

from __future__ import annotations

from typing import Any

ANCHOR_FIELDS = (
    "identity_anchors",
    "face_anchors",
    "body_anchors",
    "wardrobe_anchors",
    "style_anchors",
    "negative_prompt",
    "compact_anchor_prompt",
)
OUTFIT_FIELDS = (
    "outfit_sheet_prompt",
    "outfit_breakdown",
    "material_color_notes",
    "continuity_anchors",
    "negative_prompt",
)
POSE_FIELDS = (
    "pose_sheet_prompt",
    "pose_breakdown",
    "silhouette_notes",
    "continuity_anchors",
    "negative_prompt",
)
EXPRESSION_FIELDS = (
    "expression_sheet_prompt",
    "expression_breakdown",
    "face_continuity_anchors",
    "acting_notes",
    "negative_prompt",
)
CONTINUITY_FIELDS = (
    "continuity_score",
    "stable_anchors",
    "drift_risks",
    "conflicts",
    "missing_anchors",
    "fixed_master_prompt",
    "negative_prompt",
)


def string_object_schema(fields: tuple[str, ...], primary: str) -> dict[str, Any]:
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


ANCHOR_SCHEMA = string_object_schema(ANCHOR_FIELDS, "identity_anchors")
OUTFIT_SCHEMA = string_object_schema(OUTFIT_FIELDS, "outfit_sheet_prompt")
POSE_SCHEMA = string_object_schema(POSE_FIELDS, "pose_sheet_prompt")
EXPRESSION_SCHEMA = string_object_schema(EXPRESSION_FIELDS, "expression_sheet_prompt")
CONTINUITY_SCHEMA = string_object_schema(CONTINUITY_FIELDS, "continuity_score")
