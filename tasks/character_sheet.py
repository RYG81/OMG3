"""Structured output contract for the existing Character Sheet node."""

from __future__ import annotations

from typing import Any

CHARACTER_SHEET_FIELDS = (
    "character_summary",
    "full_body_front",
    "full_body_side",
    "full_body_back",
    "full_body_3quarter",
    "headshot_front",
    "headshot_3quarter",
    "headshot_profile",
    "expression_sheet",
    "full_body_grid",
    "face_grid",
    "master_sheet",
)

CHARACTER_SHEET_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        field: {"type": "string", **({"minLength": 1} if field == "character_summary" else {})}
        for field in CHARACTER_SHEET_FIELDS
    },
    "required": list(CHARACTER_SHEET_FIELDS),
    "additionalProperties": False,
}
