"""Schema-constrained Ollama generation with local validation and one repair attempt."""

from __future__ import annotations

import json

from ...tasks.engine import run_structured_task
from ...utils.structured_output import parse_schema
from .ollama_client import generate

_DEFAULT_SCHEMA = """{
  "type": "object",
  "properties": {
    "answer": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["answer", "confidence"],
  "additionalProperties": false
}"""


class OllamaStructuredGenerate:
    """Generate JSON constrained by a user-provided Draft 2020-12 JSON Schema."""

    CATEGORY = "ComfyUI-OMG/Core"
    FUNCTION = "generate_structured"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = (
        "validated_json",
        "raw_output",
        "is_valid",
        "validation_report",
        "provenance_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Generate schema-constrained JSON with Ollama, validate it locally, and optionally "
        "make one correction attempt when the first response is invalid."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": (
                    "STRING",
                    {
                        "default": "Answer the question and estimate confidence.",
                        "multiline": True,
                        "tooltip": "Request whose response must satisfy the JSON Schema",
                    },
                ),
                "json_schema": (
                    "STRING",
                    {
                        "default": _DEFAULT_SCHEMA,
                        "multiline": True,
                        "tooltip": "Draft 2020-12 JSON Schema object sent to Ollama and validated locally",
                    },
                ),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "Return accurate data that exactly follows the supplied JSON Schema.",
                        "multiline": True,
                    },
                ),
                "num_predict": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 8192, "step": 32},
                ),
                "temperature_override": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "-1 uses the Model Loader temperature; 0 is most reliable for schemas",
                    },
                ),
                "repair_once": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If validation fails, ask Ollama once to correct the same response",
                    },
                ),
                "cache_policy": (
                    ["use", "refresh", "bypass"],
                    {
                        "default": "use",
                        "tooltip": "Use/refresh the private process-memory cache, or bypass it entirely",
                    },
                ),
            },
        }

    def generate_structured(
        self,
        ollama_model: dict,
        prompt: str,
        json_schema: str,
        system_prompt: str = "Return accurate data that exactly follows the supplied JSON Schema.",
        num_predict: int = 1024,
        temperature_override: float = 0.0,
        repair_once: bool = True,
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        schema = parse_schema(json_schema)
        temperature = (
            float(ollama_model.get("temperature", 0.0))
            if temperature_override < 0
            else float(temperature_override)
        )
        execution = run_structured_task(
            task_id="structured_generate",
            template_version="1",
            model_profile=ollama_model,
            prompt=prompt,
            system=system_prompt,
            schema=schema,
            num_predict=num_predict,
            temperature=temperature,
            repair_once=repair_once,
            cache_policy=cache_policy,
            generate_fn=generate,
        )
        return (
            execution.structured.normalized_json,
            execution.raw_output,
            execution.structured.valid,
            execution.report,
            json.dumps(execution.provenance, indent=2, ensure_ascii=False, sort_keys=True),
        )
