"""
OllamaHandPoseHelper — Generate AI-friendly hand pose descriptions.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.design_character_tools import HAND_FIELDS, HAND_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.hand_pose_helper import (
    build_system_prompt, build_user_prompt,
    HAND_ACTION, HAND_POSITION, WHICH_HANDS, DETAIL_LEVEL,
)

_log = logging.getLogger(__name__)


class OllamaHandPoseHelper:
    """Generate detailed, AI-optimized hand pose descriptions."""
    DESCRIPTION = 'Generate detailed, AI-optimized hand pose descriptions.'

    CATEGORY = "ComfyUI-OMG/Character"
    FUNCTION = "describe_hands"
    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = (
        "hand_prompt", "right_hand", "left_hand", "finger_positions",
        "palm_orientation", "wrist_angle", "hand_interaction",
        "simplification_tips", "hand_tags", "negative_prompt", "alternative_pose",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "action": (HAND_ACTION, {"default": "relaxed open"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "action_custom": ("STRING", {"default": ""}),
                "position": (HAND_POSITION, {"default": "at sides"}),
                "position_custom": ("STRING", {"default": ""}),
                "which_hands": (WHICH_HANDS, {"default": "both hands same pose"}),
                "detail_level": (DETAIL_LEVEL, {"default": "standard - clear finger positions"}),
                "holding": ("STRING", {"default": "", "tooltip": "Object being held"}),
                "interaction": ("STRING", {"default": "", "tooltip": "Hand interaction"}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def describe_hands(self, ollama_model: dict, action: str = "relaxed open", think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        
        action_val = kwargs.get("action_custom") or action
        position_val = kwargs.get("position_custom") or kwargs.get("position", "at sides")
        
        system = build_system_prompt(
            action=action_val,
            position=position_val,
            which_hands=kwargs.get("which_hands", "both hands same pose"),
            detail_level=kwargs.get("detail_level", "standard - clear finger positions"),
            holding=kwargs.get("holding", "nothing"),
            interaction=kwargs.get("interaction", "none"),
        )
        user_prompt = build_user_prompt(kwargs.get("description", ""))
        
        _log.info("[OllamaNodes] Generating hand pose: %s", action_val)
        
        execution = run_structured_task(
            task_id="hand_pose_helper",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=HAND_SCHEMA,
            num_predict=2000,
            temperature=float(cfg.get("temperature", 0.4)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 10
        return tuple(str(parsed.get(field, "")) for field in HAND_FIELDS)
