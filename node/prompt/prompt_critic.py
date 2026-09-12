"""
OllamaPromptCritic — Prompt Critic
Analyzes prompts and provides feedback and improvements.
"""
from __future__ import annotations
import logging
from ...utils.text_utils import filter_thinking

from ...ollama_client import generate
from ...prompts.prompt_critic import build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.prompt_critic import PROMPT_CRITIC_FIELDS, PROMPT_CRITIC_SCHEMA

_log = logging.getLogger(__name__)

TARGET_MODELS = ["SDXL", "SD1.5", "Flux", "Midjourney", "DALL-E"]


class OllamaPromptCritic:
    """Analyzes prompts and provides detailed feedback."""
    DESCRIPTION = 'Analyzes prompts and provides detailed feedback.'

    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "critique"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "overall_score", "clarity_feedback", "detail_feedback", "consistency_feedback",
        "improvement_suggestions", "improved_prompt", "missing_elements", "redundant_elements",
    )
    OUTPUT_TOOLTIPS = (
        "Score out of 10 with reasoning",
        "Clarity feedback",
        "Detail level feedback",
        "Consistency/contradiction check",
        "Specific improvement suggestions",
        "Rewritten improved version",
        "What's missing",
        "What's unnecessary",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Prompt to analyze",
                }),
                "target_model": (TARGET_MODELS, {
                    "default": "SDXL",
                    "tooltip": "Target AI model",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "intended_result": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What you're trying to achieve",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def critique(self, ollama_model: dict, prompt: str,
                target_model: str = "SDXL", intended_result: str = "",
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
        system = build_system_prompt(target_model, intended_result)
        user_prompt = build_user_prompt(prompt)

        _log.info("[OllamaNodes] Critiquing prompt (target=%s)", target_model)

        execution = run_structured_task(
            task_id="prompt_critic",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=PROMPT_CRITIC_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Prompt Critic schema validation failed: %s", execution.report)
            return (execution.raw_output,) + ("",) * 7
        return tuple(str(parsed.get(field, "")) for field in PROMPT_CRITIC_FIELDS)
