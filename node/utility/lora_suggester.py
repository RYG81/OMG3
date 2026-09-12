"""
OllamaLoraSuggester — Suggest LoRAs for a prompt.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.lora_suggester import MODEL_ECOSYSTEM, build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.prompt_utility_tools import LORA_FIELDS, LORA_SCHEMA, list_or_string

_log = logging.getLogger(__name__)


class OllamaLoraSuggester:
    """Analyze a prompt and suggest helpful LoRA types."""
    DESCRIPTION = 'Analyze a prompt and suggest helpful LoRA types.'

    CATEGORY = "ComfyUI-OMG/Utility"
    FUNCTION = "suggest_loras"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "prompt_analysis", "style_loras", "concept_loras", "quality_loras",
        "priority_loras", "lora_keywords", "weight_suggestions", "prompt_optimization",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "ecosystem": (MODEL_ECOSYSTEM, {"default": "SDXL"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def suggest_loras(self, ollama_model: dict, prompt: str, ecosystem: str = "SDXL",
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
        
        system = build_system_prompt(ecosystem)
        user_prompt = build_user_prompt(prompt)
        
        _log.info("[OllamaNodes] Suggesting LoRAs for %s", ecosystem)
        
        execution = run_structured_task(
            task_id="lora_suggester",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=LORA_SCHEMA,
            num_predict=2000,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 7
        return tuple(list_or_string(parsed.get(field)) for field in LORA_FIELDS)
