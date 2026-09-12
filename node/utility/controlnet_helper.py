"""
OllamaControlNetHelper — Optimize prompts for ControlNet workflows.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.direction_prompt_tools import CONTROLNET_FIELDS, CONTROLNET_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.controlnet_helper import (
    CONTROLNET_TYPE, CONTROLNET_STRENGTH,
    build_system_prompt, build_user_prompt,
)

_log = logging.getLogger(__name__)


class OllamaControlNetHelper:
    """Optimize prompts for ControlNet workflows."""
    DESCRIPTION = 'Optimize prompts for ControlNet workflows.'

    CATEGORY = "ComfyUI-OMG/Utility"
    FUNCTION = "optimize_for_controlnet"
    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = (
        "optimized_prompt", "structural_elements", "detail_elements",
        "style_elements", "what_to_omit", "what_to_emphasize",
        "negative_prompt", "common_mistakes", "workflow_tips",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "controlnet_type": (CONTROLNET_TYPE, {"default": "canny"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "strength": (CONTROLNET_STRENGTH, {"default": "medium - 0.6-0.7"}),
                "control_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Describe the control image",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def optimize_for_controlnet(self, ollama_model: dict, prompt: str,
                                controlnet_type: str = "canny", think_mode: str = "off", filter_thinking: bool = True, **kwargs):
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
            controlnet_type=controlnet_type,
            strength=kwargs.get("strength", "medium - 0.6-0.7"),
        )
        user_prompt = build_user_prompt(prompt, kwargs.get("control_description", ""))
        
        _log.info("[OllamaNodes] Optimizing for ControlNet: %s", controlnet_type)
        
        execution = run_structured_task(
            task_id="controlnet_helper",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=CONTROLNET_SCHEMA,
            num_predict=2000,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 8
        return tuple(str(parsed.get(field, "")) for field in CONTROLNET_FIELDS)
