"""Structured synthesis contract for Image Sequence Analyzer."""

from __future__ import annotations

from typing import Any

VIDEO_SEQUENCE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "video_prompt": {"type": "string", "minLength": 1},
        "video_notes": {"type": "string"},
    },
    "required": ["video_prompt", "video_notes"],
    "additionalProperties": False,
}
