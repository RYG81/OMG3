"""
OllamaStyleTransfer — Style Transfer Prompt Generator
Extracts artistic style from an image and applies it to a new subject.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.style_transfer import build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.vision_prompt_tools import STYLE_TRANSFER_FIELDS, STYLE_TRANSFER_SCHEMA
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)


class OllamaStyleTransfer:
    """Analyzes an image's style and generates prompts to apply it to a new subject."""
    DESCRIPTION = "Analyzes an image's style and generates prompts to apply it to a new subject."

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "transfer_style"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "styled_prompt", "style_description", "style_tags",
        "color_palette", "technique_notes",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt with source style applied to the target subject",
        "Detailed description of the extracted style",
        "Comma-separated style tags",
        "Color palette from the source style",
        "Technical notes on achieving this style",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE", {"tooltip": "Image whose artistic style to extract"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "target_subject": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What to depict in the extracted style",
                }),
                "style_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly to apply the style (0=subtle, 1=dominant)",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def transfer_style(self, style_image, ollama_model: dict,
                       target_subject: str, style_strength: float = 0.7,
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
        require_declared_capability(cfg, "vision")
        img_b64 = tensor_to_base64(style_image)

        system = build_system_prompt(style_strength)
        user_prompt = build_user_prompt(target_subject)

        _log.info("[OllamaNodes] Style transfer with %s (strength=%.2f)", cfg["model"], style_strength)

        execution = run_structured_task(
            task_id="style_transfer",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=STYLE_TRANSFER_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img_b64],
            quality_context={
                "source_text": target_subject,
                "primary_field": "styled_prompt",
            },
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Style Transfer schema validation failed: %s", execution.report)
            return (execution.raw_output, "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in STYLE_TRANSFER_FIELDS)
