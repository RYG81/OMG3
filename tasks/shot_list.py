"""Typed Shot List schema, prompts, validation, and extraction helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SHOT_TARGET_MODELS = ["General Video", "Wan Video", "LTX Video"]
SEQUENCE_STYLES = [
    "single scene coverage",
    "cinematic sequence",
    "dialogue scene",
    "action sequence",
    "product showcase",
    "music video",
    "loopable sequence",
    "establishing to detail",
]


def make_shot_list_id(title: str, concept: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "shot-list"
    digest = hashlib.sha256(f"{title}\n{concept}".encode("utf-8")).hexdigest()[:10]
    return f"{slug[:40]}-{digest}"


def _shot_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "shot_id": {"type": "string", "minLength": 1},
        "order": {"type": "integer", "minimum": 1, "maximum": 24},
        "duration_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 120},
        "purpose": {"type": "string"},
        "subject_action": {"type": "string"},
        "setting": {"type": "string"},
        "time_weather": {"type": "string"},
        "framing": {"type": "string"},
        "camera_angle": {"type": "string"},
        "lens": {"type": "string"},
        "camera_motion": {"type": "string"},
        "subject_motion": {"type": "string"},
        "composition": {"type": "string"},
        "lighting": {"type": "string"},
        "transition_in": {"type": "string"},
        "transition_out": {"type": "string"},
        "continuity_locks": {"type": "array", "items": {"type": "string"}, "maxItems": 24},
        "first_frame_prompt": {"type": "string", "minLength": 1},
        "video_prompt": {"type": "string", "minLength": 1},
        "last_frame_prompt": {"type": "string", "minLength": 1},
        "negative_prompt": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_shot_list_schema(
    shot_list_id: str,
    target_model: str,
    aspect_ratio: str,
    character_ids: list[str],
    world_id: str,
    expected_shot_count: int | None,
    total_duration_seconds: float | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    continuity_properties: dict[str, Any] = {
        "character_ids": {"const": character_ids},
        "world_id": {"const": world_id},
        "global_visual_rules": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
    }
    shots: dict[str, Any] = {
        "type": "array",
        "items": _shot_schema(),
        "minItems": expected_shot_count if expected_shot_count is not None else 1,
        "maxItems": expected_shot_count if expected_shot_count is not None else 24,
    }
    properties: dict[str, Any] = {
        "shot_list_id": {"const": shot_list_id},
        "title": {"type": "string"},
        "sequence_summary": {"type": "string", "minLength": 1},
        "target_model": {"const": target_model},
        "aspect_ratio": {"const": aspect_ratio},
        "fps": (
            {"const": float(fps)}
            if fps is not None
            else {"type": "number", "minimum": 1, "maximum": 120}
        ),
        "total_duration_seconds": (
            {"const": float(total_duration_seconds)}
            if total_duration_seconds is not None
            else {"type": "number", "exclusiveMinimum": 0, "maximum": 3600}
        ),
        "continuity": {
            "type": "object",
            "properties": continuity_properties,
            "required": list(continuity_properties),
            "additionalProperties": False,
        },
        "shots": shots,
        "sequence_negative_prompt": {"type": "string"},
        "model_notes": {"type": "string"},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


SHOT_LIST_CREATE_SYSTEM = """You are a production storyboard editor and text-to-video shot planner.
Create one coherent shot list with physically plausible camera and subject motion. Every shot must
have a clear purpose, first frame, continuous video prompt, and last frame. Preserve all supplied
Character Bible IDs, World Bible ID, identity anchors, wardrobe, geography, architecture, lighting,
and visual rules. Shot IDs must be stable and orders must be sequential from 1. Durations must sum
exactly to total_duration_seconds. Prefer one continuous action per shot. Return only schema-compliant JSON."""

SHOT_LIST_UPDATE_SYSTEM = """You revise a canonical Shot List. Return the complete list, not a patch.
Never change shot_list_id, target_model, aspect_ratio, linked Character Bible IDs, or World Bible ID.
Keep shot IDs when requested. Durations must sum exactly to total_duration_seconds and shot order must
be sequential from 1. In append mode, preserve every existing shot exactly and add new shots after it.
Return only schema-compliant JSON."""


def build_create_prompt(
    shot_list_id: str,
    title: str,
    concept: str,
    target_model: str,
    shot_count: int,
    total_duration: float,
    aspect_ratio: str,
    fps: float,
    sequence_style: str,
    creative_brief: dict[str, Any] | None,
    character_bibles: list[dict[str, Any]],
    world_bible: dict[str, Any] | None,
    constraints: str,
) -> str:
    parts = [
        f"Required shot_list_id (copy exactly): {shot_list_id}",
        f"Project title: {title.strip() or 'Untitled Sequence'}",
        f"Sequence concept:\n{concept.strip()}",
        f"Target model: {target_model}",
        f"Exact shot count: {shot_count}",
        f"Exact total duration: {total_duration} seconds",
        f"Aspect ratio: {aspect_ratio}",
        f"Frame rate: {fps} fps",
        f"Sequence style: {sequence_style}",
        f"Additional constraints:\n{constraints.strip() or 'None'}",
    ]
    if creative_brief:
        parts.append("Creative Brief:\n" + json.dumps(creative_brief, indent=2, ensure_ascii=False))
    if character_bibles:
        parts.append("Linked Character Bibles:\n" + json.dumps(character_bibles, indent=2, ensure_ascii=False))
    if world_bible:
        parts.append("Linked World Bible:\n" + json.dumps(world_bible, indent=2, ensure_ascii=False))
    parts.append("Create the complete production Shot List.")
    return "\n\n".join(parts)


def build_update_prompt(data: dict[str, Any], request: str, mode: str, preserve_shot_ids: bool) -> str:
    return "\n\n".join(
        (
            f"Revision mode: {mode}",
            f"Preserve existing shot IDs: {'yes' if preserve_shot_ids else 'no'}",
            f"Requested revision:\n{request.strip()}",
            "Existing Shot List:\n" + json.dumps(data, indent=2, ensure_ascii=False),
            "Return the complete revised Shot List.",
        )
    )


def validate_shot_list(data: dict[str, Any], expected_count: int | None = None) -> list[str]:
    shots = data.get("shots")
    if not isinstance(shots, list):
        return ["shots must be a list"]
    errors = []
    if expected_count is not None and len(shots) != expected_count:
        errors.append(f"expected {expected_count} shots, received {len(shots)}")
    shot_ids = [str(shot.get("shot_id", "")) for shot in shots if isinstance(shot, dict)]
    if len(shot_ids) != len(set(shot_ids)):
        errors.append("shot_id values must be unique")
    orders = [shot.get("order") for shot in shots if isinstance(shot, dict)]
    if orders != list(range(1, len(shots) + 1)):
        errors.append("shot order must be sequential from 1")
    duration_sum = sum(float(shot.get("duration_seconds", 0)) for shot in shots if isinstance(shot, dict))
    total = float(data.get("total_duration_seconds", 0))
    if abs(duration_sum - total) > 0.05:
        errors.append(
            f"shot durations sum to {duration_sum:.3f}s but total_duration_seconds is {total:.3f}s"
        )
    return errors


def protected_update_errors(
    original: dict[str, Any], updated: dict[str, Any], mode: str, preserve_shot_ids: bool
) -> list[str]:
    protected = ["shot_list_id", "target_model", "aspect_ratio", "continuity"]
    errors = [
        f"protected field changed: {key}"
        for key in protected
        if updated.get(key) != original.get(key)
    ]
    original_shots = original.get("shots", [])
    updated_shots = updated.get("shots", [])
    if mode == "append_shots":
        if updated_shots[: len(original_shots)] != original_shots:
            errors.append("append_shots must preserve every existing shot exactly")
        if len(updated_shots) <= len(original_shots):
            errors.append("append_shots must add at least one shot")
    elif preserve_shot_ids:
        old_ids = [shot.get("shot_id") for shot in original_shots if isinstance(shot, dict)]
        new_ids = [shot.get("shot_id") for shot in updated_shots if isinstance(shot, dict)]
        if old_ids != new_ids:
            errors.append("existing shot IDs changed while preserve_shot_ids is enabled")
    return errors


def all_video_prompts(data: dict[str, Any]) -> str:
    lines = []
    for shot in data.get("shots", []):
        if isinstance(shot, dict):
            lines.append(f"{shot.get('shot_id', '')}: {shot.get('video_prompt', '')}")
    return "\n\n".join(lines)
