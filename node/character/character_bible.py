"""Typed, versioned Character Bible creation, controlled updates, and prompt adapters."""

from __future__ import annotations

import json

from ...ollama_client import generate
from ...tasks.character_bible import (
    CHARACTER_BIBLE_CREATE_SYSTEM,
    CHARACTER_BIBLE_UPDATE_SYSTEM,
    TARGET_CHARACTER_MODELS,
    VISUAL_STYLES,
    build_character_bible_schema,
    build_create_prompt,
    build_update_prompt,
    compile_character_prompt,
    copy_envelope_with_data,
    make_character_id,
    protected_update_errors,
)
from ...tasks.engine import run_structured_task


def _require_bible(envelope: dict) -> dict:
    if not isinstance(envelope, dict) or envelope.get("schema") != "omg.character_bible":
        raise ValueError("Input is not an OMG Character Bible envelope")
    if envelope.get("valid") is False:
        raise ValueError("Character Bible is marked invalid")
    data = envelope.get("data")
    if not isinstance(data, dict) or not data:
        raise ValueError("Character Bible contains no validated data")
    return data


class OllamaCharacterBibleCreate:
    """Create a validated canonical character asset for continuity workflows."""

    CATEGORY = "ComfyUI-OMG/Character/Bible"
    FUNCTION = "create_bible"
    RETURN_TYPES = (
        "OMG_CHARACTER_BIBLE",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "BOOLEAN",
        "STRING",
    )
    RETURN_NAMES = (
        "character_bible",
        "bible_json",
        "master_prompt",
        "negative_prompt",
        "compact_anchors",
        "is_valid",
        "provenance_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Create a versioned Character Bible with identity, face, body, wardrobe, style, "
        "immutable anchors, flexible traits, master prompt, and negative prompt."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "character_concept": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Canonical character concept"},
                ),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "character_name": ("STRING", {"default": ""}),
                "visual_style": (VISUAL_STYLES, {"default": "General"}),
                "target_model": (TARGET_CHARACTER_MODELS, {"default": "General"}),
                "strictness": (
                    ["balanced", "strict continuity", "production lock"],
                    {"default": "production lock"},
                ),
                "additional_constraints": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def create_bible(
        self,
        ollama_model: dict,
        character_concept: str,
        character_name: str = "",
        visual_style: str = "General",
        target_model: str = "General",
        strictness: str = "production lock",
        additional_constraints: str = "",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        if not character_concept.strip():
            raise ValueError("character_concept cannot be empty")
        character_id = make_character_id(character_name, character_concept)
        schema = build_character_bible_schema(character_id)
        execution = run_structured_task(
            task_id="character_bible_create",
            template_version="1",
            model_profile=ollama_model,
            prompt=build_create_prompt(
                character_id,
                character_name,
                character_concept,
                visual_style,
                target_model,
                strictness,
                additional_constraints,
            ),
            system=CHARACTER_BIBLE_CREATE_SYSTEM,
            schema=schema,
            num_predict=4096,
            temperature=min(float(ollama_model.get("temperature", 0.3)), 0.4),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": character_concept,
                "primary_field": "master_prompt",
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
            "schema": "omg.character_bible",
            "version": 1,
            "valid": execution.structured.valid,
            "data": data,
            "provenance": execution.provenance,
        }
        anchors = data.get("immutable_anchors", []) if data else []
        compact = "; ".join(str(item) for item in anchors) if isinstance(anchors, list) else ""
        return (
            envelope,
            json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True),
            str(data.get("master_prompt", "")),
            str(data.get("negative_prompt", "")),
            compact,
            execution.structured.valid,
            json.dumps(execution.provenance, indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaCharacterBibleUpdate:
    """Apply a controlled update while enforcing protected identity fields."""

    CATEGORY = "ComfyUI-OMG/Character/Bible"
    FUNCTION = "update_bible"
    RETURN_TYPES = ("OMG_CHARACTER_BIBLE", "STRING", "STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = (
        "updated_bible",
        "updated_bible_json",
        "master_prompt",
        "is_valid_update",
        "update_report",
        "provenance_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Update wardrobe or revise a Character Bible while enforcing character_id and optional "
        "identity-protection rules."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "character_bible": ("OMG_CHARACTER_BIBLE", {"forceInput": True}),
                "update_request": ("STRING", {"default": "", "multiline": True}),
                "update_mode": (
                    ["wardrobe_only", "controlled_revision"],
                    {"default": "wardrobe_only"},
                ),
                "preserve_identity": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "cache_policy": (["use", "refresh", "bypass"], {"default": "refresh"}),
            },
        }

    def update_bible(
        self,
        ollama_model: dict,
        character_bible: dict,
        update_request: str,
        update_mode: str,
        preserve_identity: bool,
        cache_policy: str = "refresh",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        original = _require_bible(character_bible)
        if not update_request.strip():
            raise ValueError("update_request cannot be empty")
        character_id = str(original.get("character_id", ""))
        if not character_id:
            raise ValueError("Character Bible has no character_id")
        execution = run_structured_task(
            task_id="character_bible_update",
            template_version="1",
            model_profile=ollama_model,
            prompt=build_update_prompt(original, update_request, update_mode, preserve_identity),
            system=CHARACTER_BIBLE_UPDATE_SYSTEM,
            schema=build_character_bible_schema(character_id),
            num_predict=4096,
            temperature=min(float(ollama_model.get("temperature", 0.2)), 0.3),
            repair_once=True,
            cache_policy=cache_policy,
            generate_fn=generate,
        )
        updated = execution.structured.value
        if not execution.structured.valid or not isinstance(updated, dict):
            report = f"Model output failed schema validation: {execution.report}"
            result = character_bible
            valid = False
        else:
            errors = protected_update_errors(original, updated, update_mode, preserve_identity)
            if errors:
                report = "Update rejected:\n" + "\n".join(f"- {error}" for error in errors)
                result = character_bible
                valid = False
            else:
                report = "Update accepted; protected identity fields are unchanged."
                result = copy_envelope_with_data(character_bible, updated, execution.provenance)
                valid = True
        result_data = result.get("data", {}) if isinstance(result, dict) else {}
        return (
            result,
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
            str(result_data.get("master_prompt", "")),
            valid,
            report,
            json.dumps(execution.provenance, indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaCharacterBibleToPrompt:
    """Compile a Character Bible into a scene-specific target-model prompt without an LLM call."""

    CATEGORY = "ComfyUI-OMG/Character/Bible"
    FUNCTION = "to_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "continuity_locks", "bible_json")
    DESCRIPTION = (
        "Compile a validated Character Bible into General, Flux, SDXL, Qwen, Wan, or LTX prompts."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_bible": ("OMG_CHARACTER_BIBLE", {"forceInput": True}),
                "target_model": (TARGET_CHARACTER_MODELS, {"default": "General"}),
            },
            "optional": {
                "scene_context": ("STRING", {"default": "", "multiline": True}),
                "outfit_override": ("STRING", {"default": "", "multiline": True}),
                "pose_action": ("STRING", {"default": "", "multiline": True}),
                "expression": ("STRING", {"default": ""}),
                "extra_instructions": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def to_prompt(
        self,
        character_bible: dict,
        target_model: str,
        scene_context: str = "",
        outfit_override: str = "",
        pose_action: str = "",
        expression: str = "",
        extra_instructions: str = "",
    ):
        data = _require_bible(character_bible)
        positive, negative, continuity = compile_character_prompt(
            data,
            target_model,
            scene_context,
            outfit_override,
            pose_action,
            expression,
            extra_instructions,
        )
        return (
            positive,
            negative,
            continuity,
            json.dumps(character_bible, indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaCharacterBibleInspect:
    """Expose major Character Bible sections as legacy-friendly strings."""

    CATEGORY = "ComfyUI-OMG/Character/Bible"
    FUNCTION = "inspect_bible"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "name",
        "core_identity",
        "face",
        "body_proportions",
        "default_wardrobe",
        "immutable_anchors",
        "negative_prompt",
        "master_prompt",
    )
    DESCRIPTION = "Inspect a typed Character Bible through focused string outputs."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"character_bible": ("OMG_CHARACTER_BIBLE", {"forceInput": True})}}

    def inspect_bible(self, character_bible: dict):
        data = _require_bible(character_bible)
        wardrobe = data.get("default_wardrobe", {})
        anchors = data.get("immutable_anchors", [])
        return (
            str(data.get("name", "")),
            str(data.get("core_identity", "")),
            str(data.get("face", "")),
            str(data.get("body_proportions", "")),
            json.dumps(wardrobe, ensure_ascii=False, sort_keys=True),
            "; ".join(str(item) for item in anchors) if isinstance(anchors, list) else str(anchors),
            str(data.get("negative_prompt", "")),
            str(data.get("master_prompt", "")),
        )
