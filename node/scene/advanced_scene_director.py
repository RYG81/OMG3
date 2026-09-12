"""
OllamaAdvancedSceneDirector — Advanced Scene Director
Full cinematic control with comprehensive dropdown options.
"""
from __future__ import annotations
import logging
from ...utils.text_utils import filter_thinking

from ...ollama_client import generate
from ...tasks.advanced_scene_director import ADVANCED_SCENE_FIELDS, ADVANCED_SCENE_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.advanced_scene_director import (
    build_system_prompt, build_user_prompt,
    SCENE_TYPE, CHARACTER_RELATIONSHIP, SCENE_ACTION, EMOTIONAL_TENSION,
    SCENE_COMPOSITION, ENVIRONMENT_SCALE, ENVIRONMENT_CONDITION,
    FOREGROUND_ELEMENTS, BACKGROUND_ELEMENTS,
    LIGHTING_SETUP, LIGHT_COLOR, SHADOW_INTENSITY,
    CAMERA_MOVEMENT_FEEL, LENS_EFFECTS, POST_PROCESSING,
)
from ...prompts.prompt_builder import (
    ENVIRONMENT, TIME_OF_DAY, WEATHER, CAMERA_SHOT, CAMERA_ANGLE,
    MOOD_ATMOSPHERE, ART_STYLE,
)

_log = logging.getLogger(__name__)


class OllamaAdvancedSceneDirector:
    """Advanced scene director with comprehensive dropdown options for every aspect."""
    DESCRIPTION = 'Advanced scene director with comprehensive dropdown options for every aspect.'

    CATEGORY = "ComfyUI-OMG/Scene"
    FUNCTION = "direct_scene"
    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = (
        "scene_prompt", "character_breakdown", "spatial_layout",
        "lighting_description", "camera_technical", "atmosphere_details",
        "composition_notes", "color_script", "negative_prompt", "director_vision",
    )
    OUTPUT_TOOLTIPS = (
        "Complete scene prompt",
        "Individual character descriptions",
        "Spatial arrangement details",
        "Lighting breakdown",
        "Camera specifications",
        "Atmospheric elements",
        "Composition analysis",
        "Color palette and grading",
        "Negative prompt",
        "Director's narrative intent",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "scene_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Describe what's happening in the scene",
                }),
                "num_characters": ("INT", {
                    "default": 1,
                    "min": 0, "max": 10, "step": 1,
                    "tooltip": "Number of characters (0 for landscape/still life)",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                # ═══ SCENE ═══
                "scene_type": (SCENE_TYPE, {"default": "custom", "tooltip": "Type of scene"}),
                "scene_type_custom": ("STRING", {"default": "", "tooltip": "Custom scene type"}),
                
                "relationship": (CHARACTER_RELATIONSHIP, {"default": "custom", "tooltip": "Character relationship"}),
                "relationship_custom": ("STRING", {"default": "", "tooltip": "Custom relationship"}),
                
                "scene_action": (SCENE_ACTION, {"default": "custom", "tooltip": "What's happening"}),
                "action_custom": ("STRING", {"default": "", "tooltip": "Custom action"}),
                
                "emotional_tension": (EMOTIONAL_TENSION, {"default": "custom", "tooltip": "Emotional tension level"}),
                "tension_custom": ("STRING", {"default": "", "tooltip": "Custom tension"}),
                
                # ═══ ENVIRONMENT ═══
                "environment": (ENVIRONMENT, {"default": "custom", "tooltip": "Environment/setting"}),
                "environment_custom": ("STRING", {"default": "", "tooltip": "Custom environment"}),
                
                "env_scale": (ENVIRONMENT_SCALE, {"default": "custom", "tooltip": "Environment scale"}),
                "scale_custom": ("STRING", {"default": "", "tooltip": "Custom scale"}),
                
                "env_condition": (ENVIRONMENT_CONDITION, {"default": "custom", "tooltip": "Environment condition"}),
                "condition_custom": ("STRING", {"default": "", "tooltip": "Custom condition"}),
                
                "time_of_day": (TIME_OF_DAY, {"default": "custom", "tooltip": "Time of day"}),
                "time_custom": ("STRING", {"default": "", "tooltip": "Custom time"}),
                
                "weather": (WEATHER, {"default": "custom", "tooltip": "Weather"}),
                "weather_custom": ("STRING", {"default": "", "tooltip": "Custom weather"}),
                
                "foreground": (FOREGROUND_ELEMENTS, {"default": "custom", "tooltip": "Foreground elements"}),
                "foreground_custom": ("STRING", {"default": "", "tooltip": "Custom foreground"}),
                
                "background": (BACKGROUND_ELEMENTS, {"default": "custom", "tooltip": "Background elements"}),
                "background_custom": ("STRING", {"default": "", "tooltip": "Custom background"}),
                
                # ═══ LIGHTING ═══
                "lighting_setup": (LIGHTING_SETUP, {"default": "custom", "tooltip": "Lighting setup"}),
                "lighting_custom": ("STRING", {"default": "", "tooltip": "Custom lighting setup"}),
                
                "light_color": (LIGHT_COLOR, {"default": "custom", "tooltip": "Light color"}),
                "light_color_custom": ("STRING", {"default": "", "tooltip": "Custom light color"}),
                
                "shadow_intensity": (SHADOW_INTENSITY, {"default": "custom", "tooltip": "Shadow intensity"}),
                "shadow_custom": ("STRING", {"default": "", "tooltip": "Custom shadows"}),
                
                # ═══ CAMERA ═══
                "camera_shot": (CAMERA_SHOT, {"default": "custom", "tooltip": "Camera shot type"}),
                "shot_custom": ("STRING", {"default": "", "tooltip": "Custom shot"}),
                
                "camera_angle": (CAMERA_ANGLE, {"default": "custom", "tooltip": "Camera angle"}),
                "angle_custom": ("STRING", {"default": "", "tooltip": "Custom angle"}),
                
                "camera_movement": (CAMERA_MOVEMENT_FEEL, {"default": "custom", "tooltip": "Camera movement feel"}),
                "movement_custom": ("STRING", {"default": "", "tooltip": "Custom movement"}),
                
                "lens_effects": (LENS_EFFECTS, {"default": "custom", "tooltip": "Lens effects"}),
                "lens_custom": ("STRING", {"default": "", "tooltip": "Custom lens effects"}),
                
                "post_processing": (POST_PROCESSING, {"default": "custom", "tooltip": "Post-processing/color grade"}),
                "post_custom": ("STRING", {"default": "", "tooltip": "Custom post-processing"}),
                
                # ═══ STYLE & MOOD ═══
                "mood": (MOOD_ATMOSPHERE, {"default": "custom", "tooltip": "Mood/atmosphere"}),
                "mood_custom": ("STRING", {"default": "", "tooltip": "Custom mood"}),
                
                "art_style": (ART_STYLE, {"default": "custom", "tooltip": "Art style"}),
                "style_custom": ("STRING", {"default": "", "tooltip": "Custom art style"}),
                
                "composition": (SCENE_COMPOSITION, {"default": "custom", "tooltip": "Composition style"}),
                "composition_custom": ("STRING", {"default": "", "tooltip": "Custom composition"}),
                
                # ═══ ADDITIONAL ═══
                "additional_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Any additional instructions or details",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def direct_scene(self, ollama_model: dict, scene_description: str,
                    num_characters: int = 1, think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        
        # Helper to get value (custom or dropdown)
        def get_val(dropdown_key, custom_key):
            dropdown_val = kwargs.get(dropdown_key, "custom")
            custom_val = kwargs.get(custom_key, "")
            if dropdown_val == "custom" and custom_val.strip():
                return custom_val.strip()
            elif dropdown_val != "custom":
                return dropdown_val
            return "custom"
        
        system = build_system_prompt(
            scene_type=get_val("scene_type", "scene_type_custom"),
            num_characters=num_characters,
            relationship=get_val("relationship", "relationship_custom"),
            scene_action=get_val("scene_action", "action_custom"),
            emotional_tension=get_val("emotional_tension", "tension_custom"),
            environment=get_val("environment", "environment_custom"),
            env_scale=get_val("env_scale", "scale_custom"),
            env_condition=get_val("env_condition", "condition_custom"),
            time_of_day=get_val("time_of_day", "time_custom"),
            weather=get_val("weather", "weather_custom"),
            lighting_setup=get_val("lighting_setup", "lighting_custom"),
            light_color=get_val("light_color", "light_color_custom"),
            shadow_intensity=get_val("shadow_intensity", "shadow_custom"),
            camera_shot=get_val("camera_shot", "shot_custom"),
            camera_angle=get_val("camera_angle", "angle_custom"),
            camera_movement=get_val("camera_movement", "movement_custom"),
            lens_effects=get_val("lens_effects", "lens_custom"),
            post_processing=get_val("post_processing", "post_custom"),
            mood=get_val("mood", "mood_custom"),
            art_style=get_val("art_style", "style_custom"),
            composition=get_val("composition", "composition_custom"),
            foreground=get_val("foreground", "foreground_custom"),
            background=get_val("background", "background_custom"),
            custom_instructions=kwargs.get("additional_instructions", ""),
        )
        
        user_prompt = build_user_prompt(scene_description)

        _log.info("[OllamaNodes] Advanced scene direction (%d characters)", num_characters)

        execution = run_structured_task(
            task_id="advanced_scene_director",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=ADVANCED_SCENE_SCHEMA,
            num_predict=4096,
            temperature=float(cfg.get("temperature", 0.7)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            quality_context={
                "source_text": scene_description,
                "primary_field": "scene_prompt",
            },
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Advanced Scene Director schema validation failed: %s", execution.report)
            return (execution.raw_output,) + ("",) * 9
        return tuple(str(parsed.get(field, "")) for field in ADVANCED_SCENE_FIELDS)
