"""Typed Character Bible schema, prompts, adapters, and update safeguards."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

TARGET_CHARACTER_MODELS = ["General", "Flux", "SDXL", "Qwen Image", "Wan Video", "LTX Video"]
VISUAL_STYLES = [
    "General",
    "Photoreal",
    "Cinematic",
    "Anime/Manga",
    "Illustration",
    "3D Character",
    "Comic/Graphic Novel",
]


def make_character_id(name: str, concept: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "character"
    digest = hashlib.sha256(f"{name}\n{concept}".encode("utf-8")).hexdigest()[:10]
    return f"{slug[:40]}-{digest}"


def build_character_bible_schema(character_id: str) -> dict[str, Any]:
    wardrobe = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "palette": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "materials": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "accessories": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "footwear": {"type": "string"},
        },
        "required": ["description", "palette", "materials", "accessories", "footwear"],
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {
        "character_id": {"const": character_id},
        "name": {"type": "string"},
        "core_identity": {"type": "string", "minLength": 1},
        "species": {"type": "string"},
        "age_appearance": {"type": "string"},
        "gender_presentation": {"type": "string"},
        "face": {"type": "string"},
        "hair": {"type": "string"},
        "eyes": {"type": "string"},
        "skin_surface": {"type": "string"},
        "body_proportions": {"type": "string"},
        "distinguishing_features": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 16,
        },
        "default_wardrobe": wardrobe,
        "personality_visual_cues": {"type": "string"},
        "style_anchors": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
        "immutable_anchors": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 16,
        },
        "flexible_traits": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
        "negative_prompt": {"type": "string"},
        "master_prompt": {"type": "string", "minLength": 1},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


CHARACTER_BIBLE_CREATE_SYSTEM = """You are a production character designer and continuity supervisor.
Create one canonical character bible that keeps the same identity reproducible across images,
shots, outfits, expressions, and poses. Use only details supplied or coherent details needed to
complete the design. Immutable anchors must be short, visually observable, non-overlapping facts.
Flexible traits may change without identity drift. The master prompt must be directly usable and
repeat the most important identity anchors. Return only schema-compliant JSON."""

CHARACTER_BIBLE_UPDATE_SYSTEM = """You are a character continuity supervisor updating an existing
canonical Character Bible. Return the complete updated bible, not a patch. Respect the update mode
and preserve protected identity fields. Never change character_id. If identity preservation is
requested, retain core identity, species, face, eyes, body proportions, distinguishing features,
and immutable anchors exactly. Return only schema-compliant JSON."""


def build_create_prompt(
    character_id: str,
    name: str,
    concept: str,
    visual_style: str,
    target_model: str,
    strictness: str,
    constraints: str,
) -> str:
    return "\n\n".join(
        (
            f"Required character_id (copy exactly): {character_id}",
            f"Character name: {name.strip() or 'Unnamed'}",
            f"Character concept:\n{concept.strip()}",
            f"Visual style: {visual_style}",
            f"Primary target model/family: {target_model}",
            f"Continuity strictness: {strictness}",
            f"Additional constraints:\n{constraints.strip() or 'None'}",
            "Create the complete canonical Character Bible.",
        )
    )


def build_update_prompt(
    bible_data: dict[str, Any], update_request: str, update_mode: str, preserve_identity: bool
) -> str:
    return "\n\n".join(
        (
            f"Update mode: {update_mode}",
            f"Preserve identity: {'yes' if preserve_identity else 'no'}",
            f"Requested update:\n{update_request.strip()}",
            "Existing Character Bible:\n" + json.dumps(bible_data, indent=2, ensure_ascii=False),
            "Return the complete updated Character Bible using the same character_id.",
        )
    )


def protected_update_errors(
    original: dict[str, Any], updated: dict[str, Any], update_mode: str, preserve_identity: bool
) -> list[str]:
    if updated.get("character_id") != original.get("character_id"):
        return ["character_id cannot change"]

    if update_mode == "wardrobe_only":
        allowed = {"default_wardrobe", "flexible_traits", "negative_prompt", "master_prompt"}
        protected = [key for key in original if key not in allowed]
    elif preserve_identity:
        protected = [
            "character_id",
            "core_identity",
            "species",
            "face",
            "eyes",
            "body_proportions",
            "distinguishing_features",
            "immutable_anchors",
        ]
    else:
        protected = ["character_id"]

    errors = []
    for key in protected:
        if updated.get(key) != original.get(key):
            errors.append(f"protected field changed: {key}")
    return errors


def _text(value: Any) -> str:
    return str(value or "").strip()


def _items(value: Any) -> list[str]:
    return [_text(item) for item in value] if isinstance(value, list) else []


def compile_character_prompt(
    bible_data: dict[str, Any],
    target_model: str | None = None,
    scene_context: str = "",
    outfit_override: str = "",
    pose_action: str = "",
    expression: str = "",
    extra_instructions: str = "",
) -> tuple[str, str, str]:
    target = target_model or "General"
    wardrobe = bible_data.get("default_wardrobe", {})
    wardrobe = wardrobe if isinstance(wardrobe, dict) else {}
    anchors = _items(bible_data.get("immutable_anchors"))
    continuity = "; ".join(anchors)
    wardrobe_text = outfit_override.strip() or _text(wardrobe.get("description"))
    parts = [
        _text(bible_data.get("core_identity")),
        _text(bible_data.get("face")),
        _text(bible_data.get("hair")),
        _text(bible_data.get("eyes")),
        _text(bible_data.get("skin_surface")),
        _text(bible_data.get("body_proportions")),
        ", ".join(_items(bible_data.get("distinguishing_features"))),
        wardrobe_text,
        _text(bible_data.get("personality_visual_cues")),
        scene_context.strip(),
        pose_action.strip(),
        expression.strip(),
        extra_instructions.strip(),
        f"identity continuity locks: {continuity}" if continuity else "",
    ]
    parts = [part for part in parts if part]
    if target == "SDXL":
        positive = ", ".join(parts)
    else:
        positive = ". ".join(part.rstrip(". ") for part in parts) + "."
    if target in {"Wan Video", "LTX Video"}:
        positive += " Preserve the same identity, wardrobe details, proportions, and facial structure across every frame."
    negative = _text(bible_data.get("negative_prompt"))
    return positive.strip(), negative, continuity


def copy_envelope_with_data(envelope: dict[str, Any], data: dict[str, Any], provenance: dict) -> dict:
    result = deepcopy(envelope)
    result.update(schema="omg.character_bible", version=1, valid=True)
    result["data"] = data
    result["provenance"] = provenance
    return result
