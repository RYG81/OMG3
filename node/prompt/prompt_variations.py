"""
OllamaPromptVariations — Batch Prompt Variations
Generates multiple creative variations of a prompt.
"""
from __future__ import annotations
import logging
from ...utils.text_utils import filter_thinking

from ...ollama_client import generate
from ...prompts.prompt_variations import build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.prompt_variations import VARIATION_FIELDS, build_prompt_variations_schema

_log = logging.getLogger(__name__)


class OllamaPromptVariations:
    """Generates N variations of a prompt while maintaining the core concept."""
    DESCRIPTION = 'Generates N variations of a prompt while maintaining the core concept.'

    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "generate_variations"
    RETURN_TYPES = ("STRING",) * 11
    RETURN_NAMES = (
        "variation_1", "variation_2", "variation_3", "variation_4", "variation_5",
        "variation_6", "variation_7", "variation_8", "variation_9", "variation_10",
        "all_variations",
    )
    OUTPUT_TOOLTIPS = tuple([f"Variation {i}" for i in range(1, 11)] + ["All variations newline-separated"])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "base_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Original prompt to create variations of",
                }),
                "num_variations": ("INT", {
                    "default": 5,
                    "min": 2, "max": 10, "step": 1,
                    "tooltip": "Number of variations to generate",
                }),
                "variation_strength": (["subtle", "moderate", "wild"], {
                    "default": "moderate",
                    "tooltip": "How different the variations should be",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "lock_elements": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Elements to keep unchanged (comma-separated)",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_variations(self, ollama_model: dict, base_prompt: str,
                           num_variations: int = 5, variation_strength: str = "moderate",
                           lock_elements: str = "", cache_policy: str = "use", think_mode: str = "off", filter_thinking: bool = True):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        system = build_system_prompt(num_variations, variation_strength, lock_elements)
        user_prompt = build_user_prompt(base_prompt)

        _log.info("[OllamaNodes] Generating %d variations (strength=%s)", num_variations, variation_strength)

        execution = run_structured_task(
            task_id="prompt_variations",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=build_prompt_variations_schema(num_variations),
            num_predict=4096,
            temperature=float(cfg.get("temperature", 0.8)),
            repair_once=True,
            cache_policy=cache_policy,
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Prompt Variations schema validation failed: %s", execution.report)
            return (execution.raw_output,) + ("",) * 10

        variations = [str(parsed.get(field, "")) for field in VARIATION_FIELDS]
        all_vars = str(parsed.get("all_variations") or "\n".join(value for value in variations if value))
        return tuple(variations) + (all_vars,)
