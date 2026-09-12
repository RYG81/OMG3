"""
OllamaPromptCombiner — Intelligently blend multiple prompts.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.prompt_combiner import BLEND_MODES, build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.prompt_utility_tools import COMBINER_FIELDS, COMBINER_SCHEMA

_log = logging.getLogger(__name__)


class OllamaPromptCombiner:
    """Intelligently combine multiple prompts into one cohesive prompt."""
    DESCRIPTION = 'Intelligently combine multiple prompts into one cohesive prompt.'

    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "combine"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "combined_prompt", "prompt_analysis", "conflict_resolution",
        "element_breakdown", "negative_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Final combined prompt",
        "Analysis of what each input contributed",
        "How conflicts were resolved",
        "Which elements came from which prompt",
        "Negative prompt for the result",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt_1": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "First prompt to combine",
                }),
                "blend_mode": (list(BLEND_MODES.keys()), {
                    "default": "merge",
                    "tooltip": "How to combine the prompts",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "prompt_2": ("STRING", {"default": "", "multiline": True, "tooltip": "Second prompt"}),
                "prompt_3": ("STRING", {"default": "", "multiline": True, "tooltip": "Third prompt"}),
                "prompt_4": ("STRING", {"default": "", "multiline": True, "tooltip": "Fourth prompt"}),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "weight_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def combine(self, ollama_model: dict, prompt_1: str, blend_mode: str = "merge",
                prompt_2: str = "", prompt_3: str = "", prompt_4: str = "",
                weight_1: float = 1.0, weight_2: float = 1.0,
                weight_3: float = 1.0, weight_4: float = 1.0,
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
        
        # Collect non-empty prompts
        prompts = []
        weights = []
        for p, w in [(prompt_1, weight_1), (prompt_2, weight_2), 
                     (prompt_3, weight_3), (prompt_4, weight_4)]:
            if p.strip():
                prompts.append(p.strip())
                weights.append(w)
        
        if len(prompts) < 1:
            return ("", "", "", "", "")
        
        if len(prompts) == 1:
            return (prompts[0], "Single prompt provided", "", "", "")
        
        weights_str = ", ".join([f"Prompt {i+1}: {w:.1f}" for i, w in enumerate(weights)])
        system = build_system_prompt(blend_mode, len(prompts), weights_str)
        user_prompt = build_user_prompt(prompts, weights)

        _log.info("[OllamaNodes] Combining %d prompts (mode=%s)", len(prompts), blend_mode)

        execution = run_structured_task(
            task_id="prompt_combiner",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=COMBINER_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.6)),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": "\n".join(prompts),
                "primary_field": "combined_prompt",
            },
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output, "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in COMBINER_FIELDS)
