"""Typed World Bible schema, prompts, adapters, and update safeguards."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

WORLD_TARGET_MODELS = ["General", "Flux", "SDXL", "Qwen Image", "Wan Video", "LTX Video"]
WORLD_STYLES = [
    "General",
    "Photoreal",
    "Cinematic",
    "Anime/Manga",
    "Illustration",
    "3D Environment",
    "Concept Art",
    "Graphic Novel",
]


def make_world_id(title: str, concept: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "world"
    digest = hashlib.sha256(f"{title}\n{concept}".encode("utf-8")).hexdigest()[:10]
    return f"{slug[:40]}-{digest}"


def _location_schema() -> dict[str, Any]:
    properties = {
        "name": {"type": "string", "minLength": 1},
        "function": {"type": "string"},
        "visual_identity": {"type": "string"},
        "geography": {"type": "string"},
        "architecture": {"type": "string"},
        "lighting": {"type": "string"},
        "palette": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "immutable_details": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 16,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_world_bible_schema(world_id: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "world_id": {"const": world_id},
        "title": {"type": "string"},
        "premise": {"type": "string", "minLength": 1},
        "genre_tone": {"type": "string"},
        "era_technology": {"type": "string"},
        "geography": {"type": "string"},
        "climate_weather": {"type": "string"},
        "architecture": {"type": "string"},
        "culture_society": {"type": "string"},
        "factions": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "flora_fauna": {"type": "string"},
        "materials_textures": {"type": "string"},
        "color_palette": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
        "lighting_rules": {"type": "string"},
        "visual_style": {"type": "string"},
        "key_locations": {"type": "array", "items": _location_schema(), "maxItems": 20},
        "immutable_world_rules": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 20,
        },
        "flexible_elements": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "forbidden_drift": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "negative_prompt": {"type": "string"},
        "master_environment_prompt": {"type": "string", "minLength": 1},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


WORLD_BIBLE_CREATE_SYSTEM = """You are a production world-building director, environment designer,
and continuity supervisor. Create one canonical visual World Bible for reproducible image and video
workflows. Separate immutable rules from flexible scene elements. Key locations must be visually
distinct but clearly belong to the same world. Use concrete geography, materials, architecture,
lighting, palette, culture, technology, flora, and weather details. The master environment prompt
must be directly usable. Return only schema-compliant JSON."""

WORLD_BIBLE_UPDATE_SYSTEM = """You update a canonical visual World Bible. Return the complete
updated bible, not a patch. Never change world_id. In location-expansion mode, add or refine key
locations without changing global world identity or rules. When rule preservation is enabled, keep
premise, genre/tone, era/technology, geography, architecture, culture, visual style, immutable rules,
and forbidden drift exactly unchanged. Return only schema-compliant JSON."""


def build_create_prompt(
    world_id: str,
    title: str,
    concept: str,
    visual_style: str,
    target_model: str,
    location_count: int,
    constraints: str,
) -> str:
    return "\n\n".join(
        (
            f"Required world_id (copy exactly): {world_id}",
            f"World title: {title.strip() or 'Untitled World'}",
            f"World concept:\n{concept.strip()}",
            f"Visual style: {visual_style}",
            f"Primary target model/family: {target_model}",
            f"Create approximately {location_count} key locations.",
            f"Additional continuity constraints:\n{constraints.strip() or 'None'}",
            "Create the complete canonical visual World Bible.",
        )
    )


def build_update_prompt(
    world_data: dict[str, Any], update_request: str, update_mode: str, preserve_world_rules: bool
) -> str:
    return "\n\n".join(
        (
            f"Update mode: {update_mode}",
            f"Preserve world rules: {'yes' if preserve_world_rules else 'no'}",
            f"Requested update:\n{update_request.strip()}",
            "Existing World Bible:\n" + json.dumps(world_data, indent=2, ensure_ascii=False),
            "Return the complete updated World Bible using the same world_id.",
        )
    )


def protected_update_errors(
    original: dict[str, Any], updated: dict[str, Any], update_mode: str, preserve_world_rules: bool
) -> list[str]:
    if updated.get("world_id") != original.get("world_id"):
        return ["world_id cannot change"]
    if update_mode == "location_expansion":
        allowed = {
            "key_locations",
            "flexible_elements",
            "negative_prompt",
            "master_environment_prompt",
        }
        protected = [key for key in original if key not in allowed]
    elif preserve_world_rules:
        protected = [
            "world_id",
            "premise",
            "genre_tone",
            "era_technology",
            "geography",
            "architecture",
            "culture_society",
            "visual_style",
            "immutable_world_rules",
            "forbidden_drift",
        ]
    else:
        protected = ["world_id"]
    return [f"protected field changed: {key}" for key in protected if updated.get(key) != original.get(key)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _items(value: Any) -> list[str]:
    return [_text(item) for item in value if _text(item)] if isinstance(value, list) else []


def _find_location(world_data: dict[str, Any], location_name: str) -> dict[str, Any] | None:
    needle = location_name.strip().casefold()
    if not needle:
        return None
    for location in world_data.get("key_locations", []):
        if isinstance(location, dict) and _text(location.get("name")).casefold() == needle:
            return location
    return None


def compile_world_prompt(
    world_data: dict[str, Any],
    target_model: str,
    location_name: str = "",
    scene_context: str = "",
    time_weather: str = "",
    extra_instructions: str = "",
) -> tuple[str, str, str]:
    location = _find_location(world_data, location_name)
    global_parts = [
        _text(world_data.get("premise")),
        _text(world_data.get("era_technology")),
        _text(world_data.get("geography")),
        _text(world_data.get("climate_weather")),
        _text(world_data.get("architecture")),
        _text(world_data.get("materials_textures")),
        ", ".join(_items(world_data.get("color_palette"))),
        _text(world_data.get("lighting_rules")),
        _text(world_data.get("visual_style")),
    ]
    location_parts = []
    if location:
        location_parts = [
            _text(location.get("name")),
            _text(location.get("visual_identity")),
            _text(location.get("geography")),
            _text(location.get("architecture")),
            _text(location.get("lighting")),
            ", ".join(_items(location.get("palette"))),
            "; ".join(_items(location.get("immutable_details"))),
        ]
    rules = _items(world_data.get("immutable_world_rules"))
    continuity = "; ".join(rules)
    parts = [
        *global_parts,
        *location_parts,
        scene_context.strip(),
        time_weather.strip(),
        extra_instructions.strip(),
        f"world continuity locks: {continuity}" if continuity else "",
    ]
    parts = [part for part in parts if part]
    if target_model == "SDXL":
        positive = ", ".join(parts)
    else:
        positive = ". ".join(part.rstrip(". ") for part in parts) + "."
    if target_model in {"Wan Video", "LTX Video"}:
        positive += " Preserve geography, architecture, materials, palette, and lighting rules across every frame."
    negative_parts = [_text(world_data.get("negative_prompt")), *_items(world_data.get("forbidden_drift"))]
    negative = ", ".join(dict.fromkeys(part for part in negative_parts if part))
    return positive.strip(), negative, continuity


def copy_envelope_with_data(envelope: dict[str, Any], data: dict[str, Any], provenance: dict) -> dict:
    result = deepcopy(envelope)
    result.update(schema="omg.world_bible", version=1, valid=True)
    result["data"] = data
    result["provenance"] = provenance
    return result
