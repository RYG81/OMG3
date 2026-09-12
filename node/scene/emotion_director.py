"""
OllamaEmotionDirector — Direct emotional content in prompts.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.direction_prompt_tools import EMOTION_FIELDS, EMOTION_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.emotion_director import (
    PRIMARY_EMOTIONS, EMOTION_INTENSITY, EXPRESSION_STYLE, BODY_LANGUAGE_INTENSITY,
    build_system_prompt, build_user_prompt,
)

_log = logging.getLogger(__name__)


class OllamaEmotionDirector:
    """Direct and intensify emotional content in prompts."""
    DESCRIPTION = 'Direct and intensify emotional content in prompts.'

    CATEGORY = "ComfyUI-OMG/Scene"
    FUNCTION = "direct_emotion"
    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = (
        "emotional_prompt", "facial_direction", "eye_description",
        "body_language", "color_mood", "lighting_mood",
        "environmental_mood", "emotional_tags", "avoid",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "primary_emotion": (PRIMARY_EMOTIONS, {"default": "joy"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "secondary_emotion": (["none"] + PRIMARY_EMOTIONS, {"default": "none"}),
                "intensity": (EMOTION_INTENSITY, {"default": "present - clearly visible, balanced"}),
                "expression_style": (EXPRESSION_STYLE, {"default": "realistic - natural human expression"}),
                "body_language": (BODY_LANGUAGE_INTENSITY, {"default": "moderate - clear body language"}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def direct_emotion(self, ollama_model: dict, prompt: str,
                      primary_emotion: str = "joy", think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        
        system = build_system_prompt(
            primary_emotion=primary_emotion,
            secondary_emotion=kwargs.get("secondary_emotion", "none"),
            intensity=kwargs.get("intensity", "present - clearly visible, balanced"),
            expression_style=kwargs.get("expression_style", "realistic - natural human expression"),
            body_language=kwargs.get("body_language", "moderate - clear body language"),
        )
        user_prompt = build_user_prompt(prompt)
        
        _log.info("[OllamaNodes] Directing emotion: %s", primary_emotion)
        
        execution = run_structured_task(
            task_id="emotion_director",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=EMOTION_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.7)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 8
        return tuple(str(parsed.get(field, "")) for field in EMOTION_FIELDS)
