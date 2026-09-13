"""
OllamaLightingDesigner — Design detailed lighting setups.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.design_character_tools import LIGHTING_FIELDS, LIGHTING_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.lighting_designer import (
    build_system_prompt, build_user_prompt,
    LIGHTING_SCENARIO, KEY_LIGHT_POSITION, KEY_LIGHT_QUALITY,
    FILL_LIGHT, RIM_LIGHT, PRACTICAL_LIGHTS, COLOR_TEMPERATURE,
    SHADOW_CHARACTER, VOLUMETRIC,
)
from ...prompts.prompt_builder import MOOD_ATMOSPHERE

_log = logging.getLogger(__name__)


class OllamaLightingDesigner:
    """Design detailed, production-quality lighting setups."""
    DESCRIPTION = 'Design detailed, production-quality lighting setups.'

    CATEGORY = "ComfyUI-OMG/Design"
    FUNCTION = "design_lighting"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "lighting_prompt", "technical_setup", "color_palette",
        "shadow_description", "atmosphere_effects", "mood_contribution",
        "style_tags", "avoid",
    )
    OUTPUT_TOOLTIPS = (
        "Complete lighting description for prompts",
        "Technical lighting breakdown",
        "Light colors and temperatures",
        "Shadow description",
        "Atmospheric effects",
        "How lighting affects mood",
        "Lighting tags for prompts",
        "What to avoid",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "scenario": (LIGHTING_SCENARIO, {"default": "cinematic", "tooltip": "Lighting scenario"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "scenario_custom": ("STRING", {"default": "", "tooltip": "Custom scenario"}),
                "subject": ("STRING", {"default": "", "multiline": False, "tooltip": "What's being lit"}),
                
                "key_position": (KEY_LIGHT_POSITION, {"default": "front left 45°", "tooltip": "Key light position"}),
                "key_custom": ("STRING", {"default": "", "tooltip": "Custom key position"}),
                
                "key_quality": (KEY_LIGHT_QUALITY, {"default": "soft (large diffused)", "tooltip": "Key light quality"}),
                "quality_custom": ("STRING", {"default": "", "tooltip": "Custom quality"}),
                
                "fill_light": (FILL_LIGHT, {"default": "moderate (1:2 ratio)", "tooltip": "Fill light"}),
                "fill_custom": ("STRING", {"default": "", "tooltip": "Custom fill"}),
                
                "rim_light": (RIM_LIGHT, {"default": "subtle edge", "tooltip": "Rim/back light"}),
                "rim_custom": ("STRING", {"default": "", "tooltip": "Custom rim"}),
                
                "practical": (PRACTICAL_LIGHTS, {"default": "none", "tooltip": "Practical lights"}),
                "practical_custom": ("STRING", {"default": "", "tooltip": "Custom practicals"}),
                
                "color_temp": (COLOR_TEMPERATURE, {"default": "daylight (5600K)", "tooltip": "Color temperature"}),
                "temp_custom": ("STRING", {"default": "", "tooltip": "Custom temperature"}),
                
                "shadow_char": (SHADOW_CHARACTER, {"default": "soft gradual falloff", "tooltip": "Shadow character"}),
                "shadow_custom": ("STRING", {"default": "", "tooltip": "Custom shadows"}),
                
                "volumetric": (VOLUMETRIC, {"default": "none", "tooltip": "Volumetric effects"}),
                "volumetric_custom": ("STRING", {"default": "", "tooltip": "Custom volumetric"}),
                
                "mood": (MOOD_ATMOSPHERE, {"default": "custom", "tooltip": "Mood"}),
                "mood_custom": ("STRING", {"default": "", "tooltip": "Custom mood"}),
                
                "additional": ("STRING", {"default": "", "multiline": True, "tooltip": "Additional instructions"}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def design_lighting(self, ollama_model: dict, scenario: str = "cinematic", think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        
        def get_val(dropdown_key, custom_key, default_dropdown="custom"):
            dropdown_val = kwargs.get(dropdown_key, default_dropdown)
            custom_val = kwargs.get(custom_key, "")
            if dropdown_val == "custom" and custom_val.strip():
                return custom_val.strip()
            elif dropdown_val != "custom":
                return dropdown_val
            return default_dropdown
        
        scenario_val = scenario if scenario != "custom" else kwargs.get("scenario_custom", "cinematic")
        
        system = build_system_prompt(
            scenario=scenario_val,
            key_position=get_val("key_position", "key_custom", "front left 45°"),
            key_quality=get_val("key_quality", "quality_custom", "soft (large diffused)"),
            fill_light=get_val("fill_light", "fill_custom", "moderate (1:2 ratio)"),
            rim_light=get_val("rim_light", "rim_custom", "subtle edge"),
            practical=get_val("practical", "practical_custom", "none"),
            color_temp=get_val("color_temp", "temp_custom", "daylight (5600K)"),
            shadow_char=get_val("shadow_char", "shadow_custom", "soft gradual falloff"),
            volumetric=get_val("volumetric", "volumetric_custom", "none"),
            mood=get_val("mood", "mood_custom", "cinematic"),
            custom_instructions=kwargs.get("additional", ""),
        )
        
        user_prompt = build_user_prompt(kwargs.get("subject", ""))

        _log.info("[OllamaNodes] Designing lighting for %s", scenario_val)

        execution = run_structured_task(
            task_id="lighting_designer",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=LIGHTING_SCHEMA,
            num_predict=2000,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 7
        return tuple(str(parsed.get(field, "")) for field in LIGHTING_FIELDS)
