"""
OllamaAspectOptimizer — Aspect Ratio Optimizer
Optimizes prompts for different aspect ratios.
"""
from __future__ import annotations
import logging
from ...utils.text_utils import filter_thinking

from ...ollama_client import generate
from ...prompts.aspect_optimizer import build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.prompt_utility_tools import ASPECT_FIELDS, ASPECT_SCHEMA

_log = logging.getLogger(__name__)

RATIO_OPTIONS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "2:3", "3:2"]
COMPOSITION_OPTIONS = ["subject_centered", "rule_of_thirds", "golden_ratio", "symmetrical", "dynamic"]


class OllamaAspectOptimizer:
    """Optimizes prompts for specific aspect ratios."""
    DESCRIPTION = 'Optimizes prompts for specific aspect ratios.'

    CATEGORY = "ComfyUI-OMG/Scene"
    FUNCTION = "optimize"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "optimized_prompt", "composition_notes", "cropping_suggestions", "negative_additions",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt optimized for the aspect ratio",
        "Composition adjustments made",
        "Safe zones and cropping info",
        "Additional negatives for this ratio",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Original prompt to optimize",
                }),
                "target_ratio": (RATIO_OPTIONS, {
                    "default": "16:9",
                    "tooltip": "Target aspect ratio",
                }),
                "composition_priority": (COMPOSITION_OPTIONS, {
                    "default": "rule_of_thirds",
                    "tooltip": "Composition style priority",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def optimize(self, ollama_model: dict, prompt: str,
                target_ratio: str = "16:9", composition_priority: str = "rule_of_thirds",
                cache_policy: str = "use", think_mode: str = "off", filter_thinking: bool = True):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        system = build_system_prompt(target_ratio, composition_priority)
        user_prompt = build_user_prompt(prompt)

        _log.info("[OllamaNodes] Optimizing for aspect ratio %s (composition=%s)", target_ratio, composition_priority)

        execution = run_structured_task(
            task_id="aspect_optimizer",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=ASPECT_SCHEMA,
            num_predict=2000,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Aspect Optimizer schema validation failed: %s", execution.report)
            return (execution.raw_output, "", "", "")
        return tuple(str(parsed.get(field, "")) for field in ASPECT_FIELDS)
