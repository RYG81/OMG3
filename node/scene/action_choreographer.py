"""
OllamaActionChoreographer — Design dynamic action sequences.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.direction_prompt_tools import ACTION_FIELDS, ACTION_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.action_choreographer import (
    build_system_prompt, build_user_prompt,
    ACTION_TYPE, INTENSITY_LEVEL, MOTION_PHASE, CAMERA_STYLE, PARTICIPANT_COUNT,
)
from ...prompts.prompt_builder import ENVIRONMENT, SPECIAL_EFFECTS

_log = logging.getLogger(__name__)


class OllamaActionChoreographer:
    """Choreograph dynamic action sequences and poses."""
    DESCRIPTION = 'Choreograph dynamic action sequences and poses.'

    CATEGORY = "ComfyUI-OMG/Scene"
    FUNCTION = "choreograph"
    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = (
        "action_prompt", "pose_description", "motion_direction",
        "impact_point", "facial_expressions", "clothing_dynamics",
        "environmental_interaction", "energy_visualization",
        "camera_position", "composition_tips", "negative_prompt",
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
                "action_description": ("STRING", {"default": "", "multiline": True}),
                "action_type": (ACTION_TYPE, {"default": "custom"}),
                "action_custom": ("STRING", {"default": ""}),
                "intensity": (INTENSITY_LEVEL, {"default": "energetic - dynamic motion"}),
                "motion_phase": (MOTION_PHASE, {"default": "action - mid-motion"}),
                "motion_custom": ("STRING", {"default": ""}),
                "camera_style": (CAMERA_STYLE, {"default": "dynamic angle - dramatic perspective"}),
                "camera_custom": ("STRING", {"default": ""}),
                "participants": (PARTICIPANT_COUNT, {"default": "solo - single figure"}),
                "environment": (ENVIRONMENT, {"default": "custom"}),
                "env_custom": ("STRING", {"default": ""}),
                "weapons": ("STRING", {"default": "", "tooltip": "Weapons or props involved"}),
                "effects": (SPECIAL_EFFECTS, {"default": "custom"}),
                "effects_custom": ("STRING", {"default": ""}),
                "additional": ("STRING", {"default": "", "multiline": True}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def choreograph(self, ollama_model: dict, think_mode: str = "off", filter_thinking: bool = True, **kwargs):
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
            action_type=get_val("action_type", "action_custom"),
            intensity=kwargs.get("intensity", "energetic - dynamic motion"),
            motion_phase=get_val("motion_phase", "motion_custom"),
            camera_style=get_val("camera_style", "camera_custom"),
            participants=kwargs.get("participants", "solo - single figure"),
            environment=get_val("environment", "env_custom"),
            weapons=kwargs.get("weapons", "none"),
            effects=get_val("effects", "effects_custom"),
            custom_instructions=kwargs.get("additional", ""),
        )
        user_prompt = build_user_prompt(kwargs.get("action_description", ""))
        
        _log.info("[OllamaNodes] Choreographing action")
        
        execution = run_structured_task(
            task_id="action_choreographer",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=ACTION_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.7)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            quality_context={
                "source_text": kwargs.get("action_description", ""),
                "primary_field": "action_prompt",
            },
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 10
        return tuple(str(parsed.get(field, "")) for field in ACTION_FIELDS)
