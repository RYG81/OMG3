"""
OllamaWildcardGenerator — Generate prompts with randomizable variations.
"""
from __future__ import annotations
import logging
from ...utils.text_utils import filter_thinking

from ...ollama_client import generate
from ...prompts.wildcard_generator import (
    VARIATION_STYLE, CATEGORIES_TO_VARY,
    build_system_prompt, build_user_prompt,
)
from ...tasks.engine import run_structured_task
from ...tasks.prompt_utility_tools import WILDCARD_FIELDS, WILDCARD_SCHEMA

_log = logging.getLogger(__name__)


class OllamaWildcardGenerator:
    """Generate prompts with {option1|option2|option3} wildcard syntax."""
    DESCRIPTION = 'Generate prompts with {option1|option2|option3} wildcard syntax.'

    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "generate_wildcards"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "wildcard_prompt", "static_elements", "varied_elements",
        "total_combinations", "sample_1", "sample_2", "sample_3",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt with {wildcard|syntax}",
        "Elements that stay constant",
        "What was made into wildcards",
        "Number of possible combinations",
        "Sample resolved prompt 1",
        "Sample resolved prompt 2",
        "Sample resolved prompt 3",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "base_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Base prompt to add variations to",
                }),
                "variation_style": (VARIATION_STYLE, {
                    "default": "moderate",
                    "tooltip": "How different the variations should be",
                }),
                "categories": (CATEGORIES_TO_VARY, {
                    "default": "all",
                    "tooltip": "What to vary",
                }),
                "num_options": ("INT", {
                    "default": 3,
                    "min": 2, "max": 8, "step": 1,
                    "tooltip": "Number of options per wildcard",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_wildcards(self, ollama_model: dict, base_prompt: str,
                          variation_style: str = "moderate",
                          categories: str = "all", num_options: int = 3,
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
        system = build_system_prompt(variation_style, categories, num_options)
        user_prompt = build_user_prompt(base_prompt)

        _log.info("[OllamaNodes] Generating wildcards (style=%s, options=%d)", variation_style, num_options)

        execution = run_structured_task(
            task_id="wildcard_generator",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=WILDCARD_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.8)),
            repair_once=True,
            cache_policy=cache_policy,
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 6
        samples = [str(item) for item in parsed.get("sample_outputs", [])]
        return tuple(str(parsed.get(field, "")) for field in WILDCARD_FIELDS) + tuple(samples[:3])
