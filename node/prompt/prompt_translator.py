"""
OllamaPromptTranslator — Convert prompts between model formats.
"""
from __future__ import annotations
import logging
from ...utils.text_utils import filter_thinking

from ...ollama_client import generate
from ...prompts.prompt_translator import FORMATS, build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.prompt_utility_tools import TRANSLATOR_FIELDS, TRANSLATOR_SCHEMA

_log = logging.getLogger(__name__)


class OllamaPromptTranslator:
    """Translate prompts between different AI model formats."""
    DESCRIPTION = 'Translate prompts between different AI model formats.'

    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "translate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "translated_prompt", "source_analysis", "translation_notes",
        "format_additions", "potential_issues", "negative_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt in target format",
        "Analysis of source prompt",
        "Changes made during translation",
        "Format-specific elements added",
        "Potential translation issues",
        "Negative prompt for target format",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Prompt to translate",
                }),
                "source_format": (list(FORMATS.keys()), {
                    "default": "booru_tags",
                    "tooltip": "Source prompt format",
                }),
                "target_format": (list(FORMATS.keys()), {
                    "default": "natural_flux",
                    "tooltip": "Target prompt format",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def translate(self, ollama_model: dict, prompt: str,
                 source_format: str = "booru_tags", target_format: str = "natural_flux",
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
        
        if source_format == target_format:
            return (prompt, "Same format, no translation needed", "", "", "", "")
        
        system = build_system_prompt(source_format, target_format)
        user_prompt = build_user_prompt(prompt)

        _log.info("[OllamaNodes] Translating %s → %s", source_format, target_format)

        execution = run_structured_task(
            task_id="prompt_translator",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=TRANSLATOR_SCHEMA,
            num_predict=2000,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": prompt,
                "primary_field": "translated_prompt",
                "target_model": target_format,
            },
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output, "", "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in TRANSLATOR_FIELDS)
