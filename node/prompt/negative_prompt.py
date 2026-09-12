"""
OllamaNegativePrompt — Negative Prompt Generator
Analyzes a positive prompt and generates optimal negative prompts.
"""
from __future__ import annotations
import logging
from ...utils.text_utils import filter_thinking

from ...ollama_client import generate
from ...prompts.negative_prompt import build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.negative_prompt import NEGATIVE_PROMPT_FIELDS, NEGATIVE_PROMPT_SCHEMA

_log = logging.getLogger(__name__)


class OllamaNegativePrompt:
    """Generates optimized negative prompts based on the positive prompt."""
    DESCRIPTION = 'Generates optimized negative prompts based on the positive prompt.'

    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "generate_negative"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("negative_prompt", "quality_negatives", "content_negatives", "anatomy_negatives")
    OUTPUT_TOOLTIPS = (
        "Complete negative prompt ready to use",
        "Quality-focused negatives only",
        "Content-focused negatives specific to the prompt",
        "Anatomy-focused negatives",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "positive_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The positive prompt to generate negatives for",
                }),
                "target_model": (["SDXL", "SD1.5", "Flux", "Pony", "Anime"], {
                    "default": "SDXL",
                    "tooltip": "Target AI model",
                }),
                "strictness": (["light", "balanced", "strict"], {
                    "default": "balanced",
                    "tooltip": "How comprehensive the negative prompt should be",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_negative(self, ollama_model: dict, positive_prompt: str,
                         target_model: str = "SDXL", strictness: str = "balanced",
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
        system = build_system_prompt(target_model, strictness)
        user_prompt = build_user_prompt(positive_prompt)

        _log.info("[OllamaNodes] Generating negative prompt (model=%s, strictness=%s)", target_model, strictness)

        execution = run_structured_task(
            task_id="negative_prompt",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=NEGATIVE_PROMPT_SCHEMA,
            num_predict=1024,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            generate_fn=generate,
            think=think,
            filter_thinking=filter_thinking,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Negative Prompt schema validation failed: %s", execution.report)
            raw_out = execution.raw_output
            if filter_thinking:
                raw_out = filter_thinking(raw_out, True)
            return (raw_out, "", "", "")

        # Filter thinking from outputs for proper results
        result = []
        for field in NEGATIVE_PROMPT_FIELDS:
            val = str(parsed.get(field, ""))
            if filter_thinking:
                val = filter_thinking(val, True)
            result.append(val)
        return tuple(result)
