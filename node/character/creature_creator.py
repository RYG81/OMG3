"""
OllamaCreatureCreator — Design fantasy, sci-fi, or hybrid creatures.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.design_character_tools import CREATURE_FIELDS, CREATURE_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.creature_creator import (
    build_system_prompt, build_user_prompt,
    BASE_CREATURE, SIZE_SCALE, BODY_PLAN, SURFACE_COVERING, SPECIAL_FEATURES, TEMPERAMENT,
)
from ...prompts.prompt_builder import ENVIRONMENT, COLOR_PALETTE, ART_STYLE

_log = logging.getLogger(__name__)


class OllamaCreatureCreator:
    """Design unique fantasy, sci-fi, or hybrid creatures."""
    DESCRIPTION = 'Design unique fantasy, sci-fi, or hybrid creatures.'

    CATEGORY = "ComfyUI-OMG/Character"
    FUNCTION = "create_creature"
    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = (
        "creature_prompt", "anatomy_description", "surface_details",
        "feature_details", "coloration", "eye_description",
        "movement_impression", "size_reference", "style_tags", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "creature_concept": ("STRING", {"default": "", "multiline": True}),
                "base": (BASE_CREATURE, {"default": "original design"}),
                "base_custom": ("STRING", {"default": ""}),
                "size": (SIZE_SCALE, {"default": "large - horse sized"}),
                "body_plan": (BODY_PLAN, {"default": "custom"}),
                "body_custom": ("STRING", {"default": ""}),
                "surface": (SURFACE_COVERING, {"default": "custom"}),
                "surface_custom": ("STRING", {"default": ""}),
                "features": (SPECIAL_FEATURES, {"default": "custom"}),
                "features_custom": ("STRING", {"default": ""}),
                "temperament": (TEMPERAMENT, {"default": "custom"}),
                "temperament_custom": ("STRING", {"default": ""}),
                "habitat": (ENVIRONMENT, {"default": "custom"}),
                "habitat_custom": ("STRING", {"default": ""}),
                "colors": (COLOR_PALETTE, {"default": "custom"}),
                "colors_custom": ("STRING", {"default": ""}),
                "art_style": (ART_STYLE, {"default": "custom"}),
                "style_custom": ("STRING", {"default": ""}),
                "additional_details": ("STRING", {"default": "", "multiline": True}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def create_creature(self, ollama_model: dict, think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        
        def get_val(key, custom_key):
            val = kwargs.get(key, "custom")
            custom = kwargs.get(custom_key, "")
            return custom if val == "custom" and custom.strip() else val

        system = build_system_prompt(
            base=get_val("base", "base_custom"),
            size=kwargs.get("size", "large - horse sized"),
            body_plan=get_val("body_plan", "body_custom"),
            surface=get_val("surface", "surface_custom"),
            features=get_val("features", "features_custom"),
            temperament=get_val("temperament", "temperament_custom"),
            habitat=get_val("habitat", "habitat_custom"),
            colors=get_val("colors", "colors_custom"),
            art_style=get_val("art_style", "style_custom"),
            custom_details=kwargs.get("additional_details", ""),
        )
        user_prompt = build_user_prompt(kwargs.get("creature_concept", ""))
        
        _log.info("[OllamaNodes] Creating creature")
        
        execution = run_structured_task(
            task_id="creature_creator",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=CREATURE_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.8)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            quality_context={
                "source_text": kwargs.get("creature_concept", ""),
                "primary_field": "creature_prompt",
            },
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 9
        return tuple(str(parsed.get(field, "")) for field in CREATURE_FIELDS)
