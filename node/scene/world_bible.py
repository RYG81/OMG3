"""Typed World Bible creation, protected updates, prompt compilation, and inspection."""

from __future__ import annotations

import json

from ...ollama_client import generate
from ...tasks.engine import run_structured_task
from ...tasks.world_bible import (
    WORLD_BIBLE_CREATE_SYSTEM,
    WORLD_BIBLE_UPDATE_SYSTEM,
    WORLD_STYLES,
    WORLD_TARGET_MODELS,
    build_create_prompt,
    build_update_prompt,
    build_world_bible_schema,
    compile_world_prompt,
    copy_envelope_with_data,
    make_world_id,
    protected_update_errors,
)


def _require_bible(envelope: dict) -> dict:
    if not isinstance(envelope, dict) or envelope.get("schema") != "omg.world_bible":
        raise ValueError("Input is not an OMG World Bible envelope")
    if envelope.get("valid") is False:
        raise ValueError("World Bible is marked invalid")
    data = envelope.get("data")
    if not isinstance(data, dict) or not data:
        raise ValueError("World Bible contains no validated data")
    return data


class OllamaWorldBibleCreate:
    """Create a canonical, versioned visual world asset."""

    CATEGORY = "ComfyUI-OMG/Scene/Bible"
    FUNCTION = "create_bible"
    RETURN_TYPES = (
        "OMG_WORLD_BIBLE",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "BOOLEAN",
        "STRING",
    )
    RETURN_NAMES = (
        "world_bible",
        "bible_json",
        "master_environment_prompt",
        "negative_prompt",
        "world_rules",
        "is_valid",
        "provenance_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Create a versioned World Bible with geography, culture, technology, architecture, "
        "locations, immutable rules, visual style, and environment prompts."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "world_concept": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "world_title": ("STRING", {"default": ""}),
                "visual_style": (WORLD_STYLES, {"default": "General"}),
                "target_model": (WORLD_TARGET_MODELS, {"default": "General"}),
                "location_count": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "continuity_constraints": ("STRING", {"default": "", "multiline": True}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def create_bible(
        self,
        ollama_model: dict,
        world_concept: str,
        world_title: str = "",
        visual_style: str = "General",
        target_model: str = "General",
        location_count: int = 4,
        continuity_constraints: str = "",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        if not world_concept.strip():
            raise ValueError("world_concept cannot be empty")
        world_id = make_world_id(world_title, world_concept)
        execution = run_structured_task(
            task_id="world_bible_create",
            template_version="1",
            model_profile=ollama_model,
            prompt=build_create_prompt(
                world_id,
                world_title,
                world_concept,
                visual_style,
                target_model,
                location_count,
                continuity_constraints,
            ),
            system=WORLD_BIBLE_CREATE_SYSTEM,
            schema=build_world_bible_schema(world_id),
            num_predict=6000,
            temperature=min(float(ollama_model.get("temperature", 0.3)), 0.4),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": world_concept,
                "primary_field": "master_environment_prompt",
                "target_model": target_model,
            },
            generate_fn=generate,
        )
        data = (
            execution.structured.value
            if execution.structured.valid and isinstance(execution.structured.value, dict)
            else {}
        )
        envelope = {
            "schema": "omg.world_bible",
            "version": 1,
            "valid": execution.structured.valid,
            "data": data,
            "provenance": execution.provenance,
        }
        rules = data.get("immutable_world_rules", []) if data else []
        rule_text = "; ".join(str(item) for item in rules) if isinstance(rules, list) else ""
        return (
            envelope,
            json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True),
            str(data.get("master_environment_prompt", "")),
            str(data.get("negative_prompt", "")),
            rule_text,
            execution.structured.valid,
            json.dumps(execution.provenance, indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaWorldBibleUpdate:
    """Expand locations or revise a World Bible with continuity safeguards."""

    CATEGORY = "ComfyUI-OMG/Scene/Bible"
    FUNCTION = "update_bible"
    RETURN_TYPES = ("OMG_WORLD_BIBLE", "STRING", "STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = (
        "updated_bible",
        "updated_bible_json",
        "master_environment_prompt",
        "is_valid_update",
        "update_report",
        "provenance_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = "Expand locations or revise a World Bible while protecting canonical world rules."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "world_bible": ("OMG_WORLD_BIBLE", {"forceInput": True}),
                "update_request": ("STRING", {"default": "", "multiline": True}),
                "update_mode": (
                    ["location_expansion", "controlled_revision"],
                    {"default": "location_expansion"},
                ),
                "preserve_world_rules": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "cache_policy": (["use", "refresh", "bypass"], {"default": "refresh"}),
            },
        }

    def update_bible(
        self,
        ollama_model: dict,
        world_bible: dict,
        update_request: str,
        update_mode: str,
        preserve_world_rules: bool,
        cache_policy: str = "refresh",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        original = _require_bible(world_bible)
        if not update_request.strip():
            raise ValueError("update_request cannot be empty")
        world_id = str(original.get("world_id", ""))
        if not world_id:
            raise ValueError("World Bible has no world_id")
        execution = run_structured_task(
            task_id="world_bible_update",
            template_version="1",
            model_profile=ollama_model,
            prompt=build_update_prompt(original, update_request, update_mode, preserve_world_rules),
            system=WORLD_BIBLE_UPDATE_SYSTEM,
            schema=build_world_bible_schema(world_id),
            num_predict=6000,
            temperature=min(float(ollama_model.get("temperature", 0.2)), 0.3),
            repair_once=True,
            cache_policy=cache_policy,
            generate_fn=generate,
        )
        updated = execution.structured.value
        if not execution.structured.valid or not isinstance(updated, dict):
            result = world_bible
            valid = False
            report = f"Model output failed schema validation: {execution.report}"
        else:
            errors = protected_update_errors(original, updated, update_mode, preserve_world_rules)
            if errors:
                result = world_bible
                valid = False
                report = "Update rejected:\n" + "\n".join(f"- {error}" for error in errors)
            else:
                result = copy_envelope_with_data(world_bible, updated, execution.provenance)
                valid = True
                report = "Update accepted; protected world rules are unchanged."
        result_data = result.get("data", {}) if isinstance(result, dict) else {}
        return (
            result,
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
            str(result_data.get("master_environment_prompt", "")),
            valid,
            report,
            json.dumps(execution.provenance, indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaWorldBibleToPrompt:
    """Compile a World Bible into a location-specific target prompt without an LLM call."""

    CATEGORY = "ComfyUI-OMG/Scene/Bible"
    FUNCTION = "to_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "world_continuity_locks", "bible_json")
    DESCRIPTION = "Compile a World Bible into General, Flux, SDXL, Qwen, Wan, or LTX prompts."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_bible": ("OMG_WORLD_BIBLE", {"forceInput": True}),
                "target_model": (WORLD_TARGET_MODELS, {"default": "General"}),
            },
            "optional": {
                "location_name": ("STRING", {"default": ""}),
                "scene_context": ("STRING", {"default": "", "multiline": True}),
                "time_weather": ("STRING", {"default": ""}),
                "extra_instructions": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def to_prompt(
        self,
        world_bible: dict,
        target_model: str,
        location_name: str = "",
        scene_context: str = "",
        time_weather: str = "",
        extra_instructions: str = "",
    ):
        data = _require_bible(world_bible)
        positive, negative, continuity = compile_world_prompt(
            data,
            target_model,
            location_name,
            scene_context,
            time_weather,
            extra_instructions,
        )
        return (
            positive,
            negative,
            continuity,
            json.dumps(world_bible, indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaWorldBibleInspect:
    """Expose major World Bible sections as legacy-friendly strings."""

    CATEGORY = "ComfyUI-OMG/Scene/Bible"
    FUNCTION = "inspect_bible"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "title",
        "premise",
        "era_technology",
        "geography",
        "architecture",
        "key_locations_json",
        "immutable_world_rules",
        "master_environment_prompt",
    )
    DESCRIPTION = "Inspect a typed World Bible through focused string outputs."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"world_bible": ("OMG_WORLD_BIBLE", {"forceInput": True})}}

    def inspect_bible(self, world_bible: dict):
        data = _require_bible(world_bible)
        rules = data.get("immutable_world_rules", [])
        return (
            str(data.get("title", "")),
            str(data.get("premise", "")),
            str(data.get("era_technology", "")),
            str(data.get("geography", "")),
            str(data.get("architecture", "")),
            json.dumps(data.get("key_locations", []), ensure_ascii=False, sort_keys=True),
            "; ".join(str(item) for item in rules) if isinstance(rules, list) else str(rules),
            str(data.get("master_environment_prompt", "")),
        )
