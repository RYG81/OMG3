"""
OllamaInpaintPrompt — Inpainting Prompt Generator
Analyzes context around masked area and generates fill prompts.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.inpaint_prompt import build_system_prompt, USER_PROMPT
from ...tasks.engine import run_structured_task
from ...tasks.vision_prompt_tools import INPAINT_FIELDS, INPAINT_SCHEMA
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import mask_to_base64, tensor_to_base64

_log = logging.getLogger(__name__)


class OllamaInpaintPrompt:
    """Generates inpainting prompts based on image context."""
    DESCRIPTION = 'Generates inpainting prompts based on image context.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "generate_inpaint"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("inpaint_prompt", "context_description", "negative_prompt")
    OUTPUT_TOOLTIPS = (
        "Prompt for the masked region",
        "Description of surrounding context",
        "What to avoid for seamless blending",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image with visible context"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "match_style": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match surrounding style exactly",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "mask": ("MASK", {"tooltip": "The mask indicating area to fill"}),
                "fill_intent": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What should fill the masked area (optional)",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_inpaint(self, image, ollama_model: dict, match_style: bool = True,
                        mask=None, fill_intent: str = "", cache_policy: str = "use", think_mode: str = "off", filter_thinking: bool = True):
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
        img_b64 = tensor_to_base64(image)
        system = build_system_prompt(fill_intent, match_style)
        images = [img_b64]
        prompt = USER_PROMPT
        if mask is not None:
            images.append(mask_to_base64(mask))
            prompt += (
                " Image 1 is the source image. Image 2 is the mask: white pixels are the area "
                "to fill and black pixels must remain unchanged."
            )

        _log.info("[OllamaNodes] Generating inpainting prompt (match_style=%s)", match_style)
        execution = run_structured_task(
            task_id="inpaint_prompt",
            template_version="2-mask-aware",
            model_profile=cfg,
            prompt=prompt,
            system=system,
            schema=INPAINT_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.4)),
            repair_once=True,
            cache_policy=cache_policy,
            images=images,
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Inpaint Prompt schema validation failed: %s", execution.report)
            return (execution.raw_output, "", "")
        return tuple(str(parsed.get(field, "")) for field in INPAINT_FIELDS)
