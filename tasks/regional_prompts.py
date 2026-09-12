"""Dynamic structured contracts and output normalization for Regional Prompts."""

from __future__ import annotations

from typing import Any

REGION_OUTPUT_FIELDS = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)

_GRID_REGIONS = {
    "2x2": ("top_left", "top_right", "bottom_left", "bottom_right"),
    "3x3": REGION_OUTPUT_FIELDS,
    "2x3": ("top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"),
    "3x2": ("top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right"),
    "1x3": ("left", "center", "right"),
    "3x1": ("top", "middle", "bottom"),
}
_ALL_REGION_KEYS = (*REGION_OUTPUT_FIELDS, "left", "right", "top", "middle", "bottom")


def build_regional_schema(grid_size: str) -> dict[str, Any]:
    active = _GRID_REGIONS.get(grid_size, _GRID_REGIONS["3x3"])
    properties = {key: {"type": "string"} for key in _ALL_REGION_KEYS}
    properties.update(
        {
            "full_composition": {"type": "string"},
            "regional_weights": {"type": "string"},
            "overlap_notes": {"type": "string"},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": [*active, "full_composition", "regional_weights", "overlap_notes"],
        "additionalProperties": False,
    }


def normalize_regional_outputs(parsed: dict[str, Any], grid_size: str) -> list[str]:
    values = {field: str(parsed.get(field, "") or "") for field in REGION_OUTPUT_FIELDS}
    if grid_size == "1x3":
        values["middle_left"] = str(parsed.get("left", "") or values["middle_left"])
        values["center"] = str(parsed.get("center", "") or values["center"])
        values["middle_right"] = str(parsed.get("right", "") or values["middle_right"])
    elif grid_size == "3x1":
        values["top_center"] = str(parsed.get("top", "") or values["top_center"])
        values["center"] = str(parsed.get("middle", "") or values["center"])
        values["bottom_center"] = str(parsed.get("bottom", "") or values["bottom_center"])
    return [values[field] for field in REGION_OUTPUT_FIELDS]
