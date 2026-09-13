"""Typed Creative Brief compiler and deterministic target-prompt adapter."""

from __future__ import annotations

import json

from ...tasks.creative_brief import (
    CREATIVE_BRIEF_SCHEMA,
    CREATIVE_BRIEF_SYSTEM,
    TARGET_MODELS,
    build_creative_brief_prompt,
    compile_creative_brief_prompt,
)
from ...tasks.engine import run_structured_task
from ..core.ollama_client import generate


class OllamaCreativeBriefCompiler:
    """Turn a concept and constraints into a validated, reusable production brief."""

    CATEGORY = "ComfyUI-OMG/Design"
    FUNCTION = "compile_brief"
    RETURN_TYPES = (
        "OMG_CREATIVE_BRIEF",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "BOOLEAN",
        "STRING",
    )
    RETURN_NAMES = (
        "creative_brief",
        "brief_json",
        "positive_prompt",
        "negative_prompt",
        "summary",
        "is_valid",
        "provenance_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Compile an idea into a validated, reusable creative brief and a target-model prompt pair."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "concept": (
                    "STRING",
                    {
                        "default": "A solitary explorer discovering a luminous underground forest",
                        "multiline": True,
                    },
                ),
                "target_model": (TARGET_MODELS, {"default": "General"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "aspect_ratio": (
                    ["Unspecified", "1:1", "4:3", "3:2", "16:9", "9:16", "21:9"],
                    {"default": "Unspecified"},
                ),
                "style_direction": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "constraints": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Required details, continuity locks, and things to avoid",
                    },
                ),
                "detail_level": (
                    ["concise", "standard", "production"],
                    {"default": "production"},
                ),
                "cache_policy": (
                    ["use", "refresh", "bypass"],
                    {"default": "use"},
                ),
            },
        }

    def compile_brief(
        self,
        ollama_model: dict,
        concept: str,
        target_model: str,
        aspect_ratio: str = "Unspecified",
        style_direction: str = "",
        constraints: str = "",
        detail_level: str = "production",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        prompt = build_creative_brief_prompt(
            concept,
            target_model,
            aspect_ratio,
            style_direction,
            constraints,
            detail_level,
        )
        execution = run_structured_task(
            task_id="creative_brief",
            template_version="1",
            model_profile=ollama_model,
            prompt=prompt,
            system=CREATIVE_BRIEF_SYSTEM,
            schema=CREATIVE_BRIEF_SCHEMA,
            num_predict=2048,
            temperature=min(float(ollama_model.get("temperature", 0.2)), 0.3),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": concept,
                "primary_field": "subject",
                "target_model": target_model,
                "aspect_ratio": aspect_ratio,
            },
            generate_fn=generate,
        )
        data = (
            execution.structured.value
            if execution.structured.valid and isinstance(execution.structured.value, dict)
            else {}
        )
        envelope = {
            "schema": "omg.creative_brief",
            "version": 1,
            "valid": execution.structured.valid,
            "data": data,
            "provenance": execution.provenance,
        }
        if execution.structured.valid:
            positive, negative = compile_creative_brief_prompt(data, target_model)
            summary = f"{data.get('title', 'Creative Brief')}: {data.get('intent', '')}".strip()
        else:
            positive, negative, summary = "", "", execution.report
        return (
            envelope,
            json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True),
            positive,
            negative,
            summary,
            execution.structured.valid,
            json.dumps(execution.provenance, indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaCreativeBriefToPrompt:
    """Compile a typed Creative Brief to another target family without another LLM call."""

    CATEGORY = "ComfyUI-OMG/Design"
    FUNCTION = "to_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "brief_json")
    DESCRIPTION = (
        "Deterministically adapt an OMG Creative Brief to General, Flux, SDXL, Qwen, Wan, or LTX prompting."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "creative_brief": ("OMG_CREATIVE_BRIEF", {"forceInput": True}),
                "target_model": (["Use brief target", *TARGET_MODELS], {"default": "Use brief target"}),
            },
            "optional": {
                "extra_instructions": (
                    "STRING",
                    {"default": "", "multiline": True},
                )
            },
        }

    def to_prompt(
        self,
        creative_brief: dict,
        target_model: str,
        extra_instructions: str = "",
    ):
        if not isinstance(creative_brief, dict) or creative_brief.get("schema") != "omg.creative_brief":
            raise ValueError("Input is not an OMG Creative Brief envelope")
        if creative_brief.get("valid") is False:
            raise ValueError("Creative Brief is marked invalid")
        data = creative_brief.get("data", {})
        if not isinstance(data, dict) or not data:
            raise ValueError("Creative Brief contains no validated data")
        target = None if target_model == "Use brief target" else target_model
        positive, negative = compile_creative_brief_prompt(data, target, extra_instructions)
        return (
            positive,
            negative,
            json.dumps(creative_brief, indent=2, ensure_ascii=False, sort_keys=True),
        )
