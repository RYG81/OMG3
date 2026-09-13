"""
OllamaTextureMaterial — Create detailed material and texture descriptions.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.design_character_tools import MATERIAL_FIELDS, MATERIAL_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.texture_material import (
    build_system_prompt, build_user_prompt,
    MATERIAL_CATEGORY, METAL_TYPE, FABRIC_TYPE, STONE_TYPE, WOOD_TYPE,
    SURFACE_FINISH, CONDITION,
)

_log = logging.getLogger(__name__)


class OllamaTextureMaterial:
    """Design detailed materials and textures."""
    DESCRIPTION = 'Design detailed materials and textures.'

    CATEGORY = "ComfyUI-OMG/Design"
    FUNCTION = "design_material"
    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = (
        "material_prompt", "surface_texture", "color_description",
        "reflectivity", "transparency", "pattern_details",
        "wear_details", "tactile_impression", "close_up_details",
        "material_tags", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "category": (MATERIAL_CATEGORY, {"default": "metal"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "material_description": ("STRING", {"default": "", "multiline": True}),
                "category_custom": ("STRING", {"default": ""}),
                "metal_type": (METAL_TYPE, {"default": "custom"}),
                "fabric_type": (FABRIC_TYPE, {"default": "custom"}),
                "stone_type": (STONE_TYPE, {"default": "custom"}),
                "wood_type": (WOOD_TYPE, {"default": "custom"}),
                "finish": (SURFACE_FINISH, {"default": "custom"}),
                "finish_custom": ("STRING", {"default": ""}),
                "condition": (CONDITION, {"default": "good - light use"}),
                "color": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": ""}),
                "lighting": ("STRING", {"default": ""}),
                "special": ("STRING", {"default": ""}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def design_material(self, ollama_model: dict, category: str = "metal", think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        
        # Get specific type based on category
        specific_type = "custom"
        if category == "metal":
            specific_type = kwargs.get("metal_type", "custom")
        elif category == "fabric" or category == "leather":
            specific_type = kwargs.get("fabric_type", "custom")
        elif category == "stone" or category == "ceramic":
            specific_type = kwargs.get("stone_type", "custom")
        elif category == "wood":
            specific_type = kwargs.get("wood_type", "custom")

        system = build_system_prompt(
            category=kwargs.get("category_custom") or category,
            specific_type=specific_type,
            finish=kwargs.get("finish_custom") or kwargs.get("finish", "custom"),
            condition=kwargs.get("condition", "good - light use"),
            color=kwargs.get("color", "custom"),
            pattern=kwargs.get("pattern", "custom"),
            lighting=kwargs.get("lighting", "custom"),
            special=kwargs.get("special", "none"),
        )
        user_prompt = build_user_prompt(kwargs.get("material_description", ""))
        
        _log.info("[OllamaNodes] Designing material: %s", category)
        
        execution = run_structured_task(
            task_id="texture_material",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=MATERIAL_SCHEMA,
            num_predict=2000,
            temperature=float(cfg.get("temperature", 0.6)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 10
        return tuple(str(parsed.get(field, "")) for field in MATERIAL_FIELDS)
