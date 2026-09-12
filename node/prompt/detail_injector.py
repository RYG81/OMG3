"""
OllamaDetailInjector — Add specific types of details to prompts.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.detail_injector import DETAIL_INTENSITY, build_system_prompt, build_user_prompt
from ...tasks.direction_prompt_tools import DETAIL_FIELDS, DETAIL_SCHEMA
from ...tasks.engine import run_structured_task

_log = logging.getLogger(__name__)


class OllamaDetailInjector:
    """Inject specific types of details into existing prompts."""
    DESCRIPTION = 'Inject specific types of details into existing prompts.'

    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "inject_details"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "enhanced_prompt", "added_details", "detail_tags",
        "quality_boost", "negative_additions",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "intensity": (DETAIL_INTENSITY, {"default": "moderate - noticeable but balanced"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "texture": ("BOOLEAN", {"default": True}),
                "material": ("BOOLEAN", {"default": True}),
                "lighting_details": ("BOOLEAN", {"default": False}),
                "micro_details": ("BOOLEAN", {"default": False}),
                "wear_and_tear": ("BOOLEAN", {"default": False}),
                "reflections": ("BOOLEAN", {"default": False}),
                "atmospheric": ("BOOLEAN", {"default": False}),
                "custom_detail_type": ("STRING", {"default": ""}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def inject_details(self, ollama_model: dict, prompt: str, intensity: str, think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        
        detail_types = []
        if kwargs.get("texture", True):
            detail_types.append("texture")
        if kwargs.get("material", True):
            detail_types.append("material")
        if kwargs.get("lighting_details"):
            detail_types.append("lighting_details")
        if kwargs.get("micro_details"):
            detail_types.append("micro_details")
        if kwargs.get("wear_and_tear"):
            detail_types.append("wear_and_tear")
        if kwargs.get("reflections"):
            detail_types.append("reflections")
        if kwargs.get("atmospheric"):
            detail_types.append("atmospheric")
        if kwargs.get("custom_detail_type", "").strip():
            detail_types.append(kwargs["custom_detail_type"])

        system = build_system_prompt(detail_types, intensity)
        user_prompt = build_user_prompt(prompt)
        
        _log.info("[OllamaNodes] Injecting details: %s", detail_types)
        
        execution = run_structured_task(
            task_id="detail_injector",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=DETAIL_SCHEMA,
            num_predict=2000,
            temperature=float(cfg.get("temperature", 0.6)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output, "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in DETAIL_FIELDS)
