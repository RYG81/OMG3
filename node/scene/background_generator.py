"""
OllamaBackgroundGenerator — Generate detailed backgrounds separately from subjects.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.direction_prompt_tools import BACKGROUND_FIELDS, BACKGROUND_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.background_generator import (
    build_system_prompt, build_user_prompt,
    BACKGROUND_TYPE, ARCHITECTURE_STYLE, NATURE_TYPE, URBAN_TYPE, ROOM_TYPE,
    DEPTH_COMPLEXITY, POPULATION,
)
from ...prompts.prompt_builder import TIME_OF_DAY, WEATHER, SEASON, MOOD_ATMOSPHERE, COLOR_PALETTE

_log = logging.getLogger(__name__)


class OllamaBackgroundGenerator:
    """Generate detailed background/environment descriptions."""
    DESCRIPTION = 'Generate detailed background/environment descriptions.'

    CATEGORY = "ComfyUI-OMG/Scene"
    FUNCTION = "generate_background"
    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = (
        "background_prompt", "foreground_elements", "midground_elements",
        "background_elements", "atmospheric_effects", "color_description",
        "depth_cues", "integration_tips", "negative_prompt",
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
                "description": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom description"}),
                "bg_type": (BACKGROUND_TYPE, {"default": "custom"}),
                "bg_type_custom": ("STRING", {"default": ""}),
                "architecture": (ARCHITECTURE_STYLE, {"default": "custom"}),
                "architecture_custom": ("STRING", {"default": ""}),
                "nature": (NATURE_TYPE, {"default": "custom"}),
                "nature_custom": ("STRING", {"default": ""}),
                "urban": (URBAN_TYPE, {"default": "custom"}),
                "urban_custom": ("STRING", {"default": ""}),
                "room": (ROOM_TYPE, {"default": "custom"}),
                "room_custom": ("STRING", {"default": ""}),
                "depth": (DEPTH_COMPLEXITY, {"default": "medium - foreground and background"}),
                "population": (POPULATION, {"default": "empty - no one"}),
                "time_of_day": (TIME_OF_DAY, {"default": "custom"}),
                "time_custom": ("STRING", {"default": ""}),
                "weather": (WEATHER, {"default": "custom"}),
                "weather_custom": ("STRING", {"default": ""}),
                "season": (SEASON, {"default": "custom"}),
                "season_custom": ("STRING", {"default": ""}),
                "mood": (MOOD_ATMOSPHERE, {"default": "custom"}),
                "mood_custom": ("STRING", {"default": ""}),
                "colors": (COLOR_PALETTE, {"default": "custom"}),
                "colors_custom": ("STRING", {"default": ""}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_background(self, ollama_model: dict, think_mode: str = "off", filter_thinking: bool = True, **kwargs):
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
            bg_type=get_val("bg_type", "bg_type_custom"),
            architecture=get_val("architecture", "architecture_custom"),
            nature=get_val("nature", "nature_custom"),
            urban=get_val("urban", "urban_custom"),
            room=get_val("room", "room_custom"),
            depth=kwargs.get("depth", "medium - foreground and background"),
            population=kwargs.get("population", "empty - no one"),
            time_of_day=get_val("time_of_day", "time_custom"),
            weather=get_val("weather", "weather_custom"),
            season=get_val("season", "season_custom"),
            mood=get_val("mood", "mood_custom"),
            colors=get_val("colors", "colors_custom"),
        )

        user_prompt = build_user_prompt(kwargs.get("description", ""))
        
        _log.info("[OllamaNodes] Generating background")
        
        execution = run_structured_task(
            task_id="background_generator",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=BACKGROUND_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.7)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 8
        return tuple(str(parsed.get(field, "")) for field in BACKGROUND_FIELDS)
