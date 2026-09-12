"""Typed Shot List creation, protected revision, shot extraction, and inspection."""

from __future__ import annotations

import json

from ...ollama_client import generate
from ...tasks.engine import run_structured_task
from ...tasks.shot_list import (
    SEQUENCE_STYLES,
    SHOT_LIST_CREATE_SYSTEM,
    SHOT_LIST_UPDATE_SYSTEM,
    SHOT_TARGET_MODELS,
    all_video_prompts,
    build_create_prompt,
    build_shot_list_schema,
    build_update_prompt,
    make_shot_list_id,
    protected_update_errors,
    validate_shot_list,
)


def _require_envelope(envelope: dict, schema_name: str) -> dict:
    if not isinstance(envelope, dict) or envelope.get("schema") != schema_name:
        raise ValueError(f"Input is not a {schema_name} envelope")
    if envelope.get("valid") is False:
        raise ValueError(f"{schema_name} is marked invalid")
    data = envelope.get("data")
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{schema_name} contains no validated data")
    return data


def _optional_data(envelope, schema_name: str) -> dict | None:
    if envelope is None:
        return None
    return _require_envelope(envelope, schema_name)


def _shot_result(
    data: dict,
    provenance: dict,
    valid: bool,
    report: str,
    envelope_override: dict | None = None,
) -> tuple:
    envelope = envelope_override or {
        "schema": "omg.shot_list",
        "version": 1,
        "valid": valid,
        "data": data if valid else {},
        "provenance": provenance,
    }
    exposed_data = data if valid or envelope_override is not None else {}
    return (
        envelope,
        json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True),
        all_video_prompts(exposed_data),
        str(exposed_data.get("sequence_negative_prompt", "")),
        len(exposed_data.get("shots", [])) if isinstance(exposed_data.get("shots"), list) else 0,
        valid,
        report,
        json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True),
    )


class OllamaShotListCreate:
    """Create a typed production shot list linked to optional creative/world/character assets."""

    CATEGORY = "ComfyUI-OMG/Video/Shot List"
    FUNCTION = "create_shot_list"
    RETURN_TYPES = (
        "OMG_SHOT_LIST",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "BOOLEAN",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "shot_list",
        "shot_list_json",
        "all_video_prompts",
        "sequence_negative_prompt",
        "shot_count",
        "is_valid",
        "validation_report",
        "provenance_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Create a validated shot list with duration, first/video/last-frame prompts, motion, "
        "camera, lighting, transitions, and linked continuity assets."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "sequence_concept": ("STRING", {"default": "", "multiline": True}),
                "target_model": (SHOT_TARGET_MODELS, {"default": "Wan Video"}),
                "shot_count": ("INT", {"default": 5, "min": 1, "max": 12, "step": 1}),
                "total_duration_seconds": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.5, "max": 300.0, "step": 0.5},
                ),
                "aspect_ratio": (
                    ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    {"default": "16:9"},
                ),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "project_title": ("STRING", {"default": ""}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "sequence_style": (SEQUENCE_STYLES, {"default": "cinematic sequence"}),
                "creative_brief": ("OMG_CREATIVE_BRIEF", {"forceInput": True}),
                "character_bible_1": ("OMG_CHARACTER_BIBLE", {"forceInput": True}),
                "character_bible_2": ("OMG_CHARACTER_BIBLE", {"forceInput": True}),
                "character_bible_3": ("OMG_CHARACTER_BIBLE", {"forceInput": True}),
                "world_bible": ("OMG_WORLD_BIBLE", {"forceInput": True}),
                "additional_constraints": ("STRING", {"default": "", "multiline": True}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def create_shot_list(
        self,
        ollama_model: dict,
        sequence_concept: str,
        target_model: str,
        shot_count: int,
        total_duration_seconds: float,
        aspect_ratio: str,
        project_title: str = "",
        fps: float = 24.0,
        sequence_style: str = "cinematic sequence",
        creative_brief=None,
        character_bible_1=None,
        character_bible_2=None,
        character_bible_3=None,
        world_bible=None,
        additional_constraints: str = "",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        if not sequence_concept.strip():
            raise ValueError("sequence_concept cannot be empty")
        creative_data = _optional_data(creative_brief, "omg.creative_brief")
        character_data = [
            _require_envelope(item, "omg.character_bible")
            for item in (character_bible_1, character_bible_2, character_bible_3)
            if item is not None
        ]
        world_data = _optional_data(world_bible, "omg.world_bible")
        character_ids = [str(item.get("character_id", "")) for item in character_data]
        world_id = str(world_data.get("world_id", "")) if world_data else ""
        shot_list_id = make_shot_list_id(project_title, sequence_concept)
        execution = run_structured_task(
            task_id="shot_list_create",
            template_version="1",
            model_profile=ollama_model,
            prompt=build_create_prompt(
                shot_list_id,
                project_title,
                sequence_concept,
                target_model,
                shot_count,
                total_duration_seconds,
                aspect_ratio,
                fps,
                sequence_style,
                creative_data,
                character_data,
                world_data,
                additional_constraints,
            ),
            system=SHOT_LIST_CREATE_SYSTEM,
            schema=build_shot_list_schema(
                shot_list_id,
                target_model,
                aspect_ratio,
                character_ids,
                world_id,
                shot_count,
                total_duration_seconds,
                fps,
            ),
            num_predict=7000,
            temperature=min(float(ollama_model.get("temperature", 0.3)), 0.4),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "expected_shot_count": shot_count,
                "expected_total_duration": float(total_duration_seconds),
                "expected_target_model": target_model,
                "expected_aspect_ratio": aspect_ratio,
                "expected_fps": float(fps),
                "expected_character_ids": character_ids,
                "expected_world_id": world_id,
            },
            generate_fn=generate,
        )
        value = execution.structured.value
        errors = (
            validate_shot_list(value, shot_count)
            if execution.structured.valid and isinstance(value, dict)
            else [execution.report]
        )
        provenance = dict(execution.provenance)
        provenance["domain_valid"] = not errors
        if errors:
            provenance["domain_errors"] = errors
        report = "Shot List is schema-valid and internally consistent." if not errors else "\n".join(errors)
        return _shot_result(value if isinstance(value, dict) else {}, provenance, not errors, report)


class OllamaShotListUpdate:
    """Revise, append, or resequence shots while preserving linked continuity assets."""

    CATEGORY = "ComfyUI-OMG/Video/Shot List"
    FUNCTION = "update_shot_list"
    RETURN_TYPES = OllamaShotListCreate.RETURN_TYPES
    RETURN_NAMES = OllamaShotListCreate.RETURN_NAMES
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Revise or append shots while protecting shot-list ID, target model, aspect ratio, "
        "Character Bible links, and World Bible link."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "shot_list": ("OMG_SHOT_LIST", {"forceInput": True}),
                "revision_request": ("STRING", {"default": "", "multiline": True}),
                "revision_mode": (
                    ["revise_existing", "append_shots", "resequence"],
                    {"default": "revise_existing"},
                ),
                "preserve_shot_ids": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "cache_policy": (["use", "refresh", "bypass"], {"default": "refresh"}),
            },
        }

    def update_shot_list(
        self,
        ollama_model: dict,
        shot_list: dict,
        revision_request: str,
        revision_mode: str,
        preserve_shot_ids: bool,
        cache_policy: str = "refresh",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        original = _require_envelope(shot_list, "omg.shot_list")
        if not revision_request.strip():
            raise ValueError("revision_request cannot be empty")
        continuity = original.get("continuity", {})
        character_ids = continuity.get("character_ids", []) if isinstance(continuity, dict) else []
        world_id = str(continuity.get("world_id", "")) if isinstance(continuity, dict) else ""
        execution = run_structured_task(
            task_id="shot_list_update",
            template_version="1",
            model_profile=ollama_model,
            prompt=build_update_prompt(original, revision_request, revision_mode, preserve_shot_ids),
            system=SHOT_LIST_UPDATE_SYSTEM,
            schema=build_shot_list_schema(
                str(original.get("shot_list_id", "")),
                str(original.get("target_model", "")),
                str(original.get("aspect_ratio", "")),
                list(character_ids),
                world_id,
                None,
                (
                    None
                    if revision_mode == "append_shots"
                    else float(original.get("total_duration_seconds", 0))
                ),
                float(original.get("fps", 24)),
            ),
            num_predict=7000,
            temperature=min(float(ollama_model.get("temperature", 0.2)), 0.3),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "expected_shot_count": None,
                "expected_total_duration": (
                    None
                    if revision_mode == "append_shots"
                    else float(original.get("total_duration_seconds", 0))
                ),
                "expected_target_model": str(original.get("target_model", "")),
                "expected_aspect_ratio": str(original.get("aspect_ratio", "")),
                "expected_fps": float(original.get("fps", 24)),
                "expected_character_ids": list(character_ids),
                "expected_world_id": world_id,
                "revision_mode": revision_mode,
                "preserve_shot_ids": preserve_shot_ids,
                "original_shots": original.get("shots", []),
            },
            generate_fn=generate,
        )
        updated = execution.structured.value
        errors = []
        if not execution.structured.valid or not isinstance(updated, dict):
            errors.append(execution.report)
        else:
            errors.extend(validate_shot_list(updated))
            errors.extend(
                protected_update_errors(original, updated, revision_mode, preserve_shot_ids)
            )
        provenance = dict(execution.provenance)
        provenance["domain_valid"] = not errors
        if errors:
            provenance["domain_errors"] = errors
            report = "Update rejected:\n" + "\n".join(f"- {error}" for error in errors)
            return _shot_result(original, provenance, False, report, envelope_override=shot_list)
        report = "Shot List update accepted; linked continuity assets are unchanged."
        return _shot_result(updated, provenance, True, report)


class OllamaShotListGetShot:
    """Extract one shot as focused strings for downstream image/video nodes."""

    CATEGORY = "ComfyUI-OMG/Video/Shot List"
    FUNCTION = "get_shot"
    RETURN_TYPES = (
        "STRING",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "shot_id",
        "duration_seconds",
        "purpose",
        "first_frame_prompt",
        "video_prompt",
        "last_frame_prompt",
        "negative_prompt",
        "camera_direction",
        "lighting",
        "continuity_locks",
        "shot_json",
    )
    DESCRIPTION = "Extract one indexed shot with frame prompts, motion, camera, lighting, and locks."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_list": ("OMG_SHOT_LIST", {"forceInput": True}),
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 24, "step": 1}),
            }
        }

    def get_shot(self, shot_list: dict, shot_index: int):
        data = _require_envelope(shot_list, "omg.shot_list")
        shots = data.get("shots", [])
        if not isinstance(shots, list) or not shots:
            raise ValueError("Shot List contains no shots")
        index = max(1, min(int(shot_index), len(shots)))
        shot = shots[index - 1]
        if not isinstance(shot, dict):
            raise ValueError("Selected shot is invalid")
        camera = ", ".join(
            str(shot.get(key, ""))
            for key in ("framing", "camera_angle", "lens", "camera_motion", "composition")
            if shot.get(key)
        )
        locks = shot.get("continuity_locks", [])
        lock_text = "; ".join(str(item) for item in locks) if isinstance(locks, list) else str(locks)
        return (
            str(shot.get("shot_id", "")),
            float(shot.get("duration_seconds", 0)),
            str(shot.get("purpose", "")),
            str(shot.get("first_frame_prompt", "")),
            str(shot.get("video_prompt", "")),
            str(shot.get("last_frame_prompt", "")),
            str(shot.get("negative_prompt", "")),
            camera,
            str(shot.get("lighting", "")),
            lock_text,
            json.dumps(shot, indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaShotListInspect:
    """Inspect sequence-level metadata and a compact timeline."""

    CATEGORY = "ComfyUI-OMG/Video/Shot List"
    FUNCTION = "inspect_shot_list"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "title",
        "sequence_summary",
        "target_model",
        "total_duration_seconds",
        "shot_count",
        "character_ids",
        "world_id",
        "timeline_json",
    )
    DESCRIPTION = "Inspect a typed Shot List through sequence metadata and compact timeline JSON."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shot_list": ("OMG_SHOT_LIST", {"forceInput": True})}}

    def inspect_shot_list(self, shot_list: dict):
        data = _require_envelope(shot_list, "omg.shot_list")
        shots = data.get("shots", [])
        continuity = data.get("continuity", {})
        timeline = [
            {
                "shot_id": shot.get("shot_id", ""),
                "order": shot.get("order", 0),
                "duration_seconds": shot.get("duration_seconds", 0),
                "purpose": shot.get("purpose", ""),
                "transition_out": shot.get("transition_out", ""),
            }
            for shot in shots
            if isinstance(shot, dict)
        ]
        character_ids = continuity.get("character_ids", []) if isinstance(continuity, dict) else []
        return (
            str(data.get("title", "")),
            str(data.get("sequence_summary", "")),
            str(data.get("target_model", "")),
            float(data.get("total_duration_seconds", 0)),
            len(shots) if isinstance(shots, list) else 0,
            ", ".join(str(item) for item in character_ids),
            str(continuity.get("world_id", "")) if isinstance(continuity, dict) else "",
            json.dumps(timeline, indent=2, ensure_ascii=False),
        )
