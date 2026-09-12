"""
OllamaPromptEnhancer — Prompt Enhancer
Takes a simple prompt and enhances it with rich details.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.prompt_enhance import SYSTEM_PROMPT_PRESETS, build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.prompt_enhancer import PROMPT_ENHANCER_SCHEMA
from ...utils.system_prompt import compose_system_prompt
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)


class OllamaPromptEnhancer:
    """Enhances simple prompts into detailed, optimized prompts for AI image generation."""
    DESCRIPTION = 'Enhances simple prompts into detailed, optimized prompts for AI image generation.'

    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt", "tags", "changes_summary")
    OUTPUT_TOOLTIPS = (
        "Enhanced, optimized prompt ready for use",
        "Suggested negative prompt",
        "Extracted tags/keywords",
        "Summary of changes made",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "simple_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Your basic prompt to enhance",
                }),
                "system_prompt_preset": (list(SYSTEM_PROMPT_PRESETS), {
                    "default": "default",
                    "tooltip": "System prompt preset to use for enhancement",
                }),
                "enhancement_level": (["subtle", "moderate", "dramatic"], {
                    "default": "moderate",
                    "tooltip": "How much to enhance the prompt",
                }),
                "target_model": (["SDXL", "SD1.5", "Flux", "Midjourney"], {
                    "default": "SDXL",
                    "tooltip": "Target AI model to optimize the prompt for",
                }),
                "aspect_focus": (["balanced", "detail", "mood", "technical"], {
                    "default": "balanced",
                    "tooltip": "What aspect to focus the enhancement on",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def enhance(
        self,
        ollama_model: dict,
        simple_prompt: str,
        system_prompt_preset: str = "default",
        enhancement_level: str = "moderate",
        target_model: str = "SDXL",
        aspect_focus: str = "balanced",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
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
            enhancement_level, target_model, aspect_focus, system_prompt_preset
        )
        user_prompt = build_user_prompt(simple_prompt, system_prompt_preset)
        _log.info(
            "Enhancing prompt with %s (level=%s, target=%s)",
            cfg["model"],
            enhancement_level,
            target_model,
        )

        # Preserve the legacy free-form PromptForge preset behavior. The default
        # preset now uses the shared schema/task engine.
        if system_prompt_preset == "2026U":
            raw = generate(
                base_url=cfg["base_url"],
                model=cfg["model"],
                prompt=user_prompt,
                system=compose_system_prompt(system, cfg, think=think, filter_thinking=filter_thinking),
                temperature=cfg.get("temperature", 0.7),
                num_ctx=cfg.get("num_ctx", 8192),
                num_predict=2048,
                seed=cfg.get("seed", -1),
                keep_alive=cfg.get("keep_alive", "5m"),
                think=think,
                filter_thinking=filter_thinking,
            )
            if filter_thinking:
                raw = filter_thinking(raw, True)
            return (raw, "", "", "")

        execution = run_structured_task(
            task_id="prompt_enhancer",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=PROMPT_ENHANCER_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.7)),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": simple_prompt,
                "primary_field": "enhanced_prompt",
                "target_model": target_model,
            },
            generate_fn=generate,
            think=think,
            filter_thinking=filter_thinking,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Prompt enhancement schema validation failed: %s", execution.report)
            raw_out = execution.raw_output
            if filter_thinking:
                raw_out = filter_thinking(raw_out, True)
            return (raw_out, "", "", "")

        enhanced = str(parsed.get("enhanced_prompt", ""))
        negative = str(parsed.get("negative_prompt", ""))
        tags = str(parsed.get("tags", ""))
        changes = str(parsed.get("changes_summary", ""))

        if filter_thinking:
            enhanced = filter_thinking(enhanced, True)
            negative = filter_thinking(negative, True)
            tags = filter_thinking(tags, True)
            changes = filter_thinking(changes, True)

        return (enhanced, negative, tags, changes)
