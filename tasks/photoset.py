"""Structured synthesis contract for Photoset Folder Analyzer."""

from __future__ import annotations

from typing import Any

PHOTOSET_FIELDS = (
    "photoset_description",
    "consistent_subject_identity",
    "consistent_clothing",
    "consistent_environment",
    "consistent_camera_lighting_style",
    "variable_elements",
    "consistent_prompt",
    "negative_prompt",
)

PHOTOSET_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        field: {"type": "string", **({"minLength": 1} if field == "consistent_prompt" else {})}
        for field in PHOTOSET_FIELDS
    },
    "required": list(PHOTOSET_FIELDS),
    "additionalProperties": False,
}
