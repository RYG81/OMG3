"""
OllamaStyleIdentifier — Art Style Identifier
Identifies art styles, movements, and techniques in images.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.style_identifier import build_system_prompt, USER_PROMPT
from ...tasks.engine import run_structured_task
from ...tasks.style_identifier import STYLE_IDENTIFIER_SCHEMA, STYLE_OUTPUT_FIELDS
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)

DEPTH_OPTIONS = ["quick", "detailed", "comprehensive"]


class OllamaStyleIdentifier:
    """Identifies artistic styles, movements, and techniques."""
    DESCRIPTION = 'Identifies artistic styles, movements, and techniques.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "identify_style"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "primary_style", "style_movement", "techniques", "possible_influences",
        "medium", "style_tags", "similar_artists", "replication_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Main art style identified",
        "Art movement/era",
        "Artistic techniques used",
        "Artist/work influences",
        "Apparent medium",
        "Tags for replication",
        "Artists with similar style",
        "Prompt to recreate this style",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to analyze"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "depth": (DEPTH_OPTIONS, {
                    "default": "detailed",
                    "tooltip": "Analysis depth",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def identify_style(
        self, image, ollama_model: dict, depth: str = "detailed", cache_policy: str = "use"
    , think_mode: str = "off", filter_thinking: bool = True):
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
        system = build_system_prompt(depth)

        _log.info("[OllamaNodes] Identifying art style (depth=%s)", depth)

        execution = run_structured_task(
            task_id="style_identifier",
            template_version="1",
            model_profile=cfg,
            prompt=USER_PROMPT,
            system=system,
            schema=STYLE_IDENTIFIER_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.4)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Style Identifier schema validation failed: %s", execution.report)
            return (execution.raw_output,) + ("",) * 7
        return tuple(str(parsed.get(field, "")) for field in STYLE_OUTPUT_FIELDS)
