"""Typed prompt/image evaluation, comparison, and inspection nodes."""

from __future__ import annotations

import json

from ...ollama_client import generate
from ...tasks.engine import run_structured_task
from ...tasks.evaluation import (
    EVALUATION_FOCUS,
    EVALUATION_TARGETS,
    IMAGE_EVALUATOR_SYSTEM,
    PROMPT_EVALUATOR_SYSTEM,
    build_evaluation_schema,
    build_image_evaluation_prompt,
    build_prompt_evaluation_prompt,
    compact_asset_context,
    make_evaluation_envelope,
    normalize_evaluation,
    require_evaluation,
)
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64


def _result(execution, data: dict, valid: bool, report: str):
    envelope = make_evaluation_envelope(data, execution.provenance, valid)
    strengths = data.get("strengths", []) if valid else []
    issues = data.get("issues", []) if valid else []
    return (
        envelope,
        json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True),
        float(data.get("overall_score", 0)) if valid else 0.0,
        str(data.get("verdict", "invalid")) if valid else "invalid",
        "\n".join(str(item) for item in strengths),
        json.dumps(issues, indent=2, ensure_ascii=False),
        str(data.get("improved_prompt", "")) if valid else "",
        valid,
        report,
        json.dumps(execution.provenance, indent=2, ensure_ascii=False, sort_keys=True),
    )


class OllamaPromptEvaluator:
    """Evaluate a prompt against intent, model needs, and optional typed continuity assets."""

    CATEGORY = "ComfyUI-OMG/Evaluation"
    FUNCTION = "evaluate_prompt"
    RETURN_TYPES = (
        "OMG_EVALUATION",
        "STRING",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "BOOLEAN",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "evaluation",
        "evaluation_json",
        "overall_score",
        "verdict",
        "strengths",
        "issues_json",
        "improved_prompt",
        "is_valid",
        "validation_report",
        "provenance_json",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Evaluate a generation prompt with weighted criteria and optional Creative Brief, "
        "Character Bible, World Bible, and Shot List requirements."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "target_model": (EVALUATION_TARGETS, {"default": "General"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "intended_result": ("STRING", {"default": "", "multiline": True}),
                "evaluation_focus": (EVALUATION_FOCUS, {"default": "balanced"}),
                "creative_brief": ("OMG_CREATIVE_BRIEF", {"forceInput": True}),
                "character_bible": ("OMG_CHARACTER_BIBLE", {"forceInput": True}),
                "world_bible": ("OMG_WORLD_BIBLE", {"forceInput": True}),
                "shot_list": ("OMG_SHOT_LIST", {"forceInput": True}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def evaluate_prompt(
        self,
        ollama_model: dict,
        prompt: str,
        target_model: str,
        intended_result: str = "",
        evaluation_focus: str = "balanced",
        creative_brief=None,
        character_bible=None,
        world_bible=None,
        shot_list=None,
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        context = compact_asset_context(
            creative_brief, character_bible, world_bible, shot_list
        )
        execution = run_structured_task(
            task_id="prompt_evaluator",
            template_version="1",
            model_profile=ollama_model,
            prompt=build_prompt_evaluation_prompt(
                prompt, target_model, intended_result, evaluation_focus, context
            ),
            system=PROMPT_EVALUATOR_SYSTEM,
            schema=build_evaluation_schema("prompt", target_model),
            num_predict=4096,
            temperature=min(float(ollama_model.get("temperature", 0.2)), 0.3),
            repair_once=True,
            cache_policy=cache_policy,
            generate_fn=generate,
        )
        value = execution.structured.value
        if not execution.structured.valid or not isinstance(value, dict):
            return _result(execution, {}, False, execution.report)
        data = normalize_evaluation(value, execution.provenance)
        return _result(execution, data, True, "Evaluation is schema-valid; score and verdict were recomputed locally.")


class OllamaImageEvaluator:
    """Evaluate a generated image against prompt, reference, and typed continuity assets."""

    CATEGORY = "ComfyUI-OMG/Evaluation"
    FUNCTION = "evaluate_image"
    RETURN_TYPES = OllamaPromptEvaluator.RETURN_TYPES
    RETURN_NAMES = OllamaPromptEvaluator.RETURN_NAMES
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Evaluate a candidate image against source prompt, optional reference image, and typed "
        "creative/character/world/shot requirements."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ollama_model": ("OLLAMA_MODEL",),
                "target_model": (EVALUATION_TARGETS, {"default": "General"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "source_prompt": ("STRING", {"default": "", "multiline": True}),
                "evaluation_focus": (EVALUATION_FOCUS, {"default": "balanced"}),
                "creative_brief": ("OMG_CREATIVE_BRIEF", {"forceInput": True}),
                "character_bible": ("OMG_CHARACTER_BIBLE", {"forceInput": True}),
                "world_bible": ("OMG_WORLD_BIBLE", {"forceInput": True}),
                "shot_list": ("OMG_SHOT_LIST", {"forceInput": True}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def evaluate_image(
        self,
        image,
        ollama_model: dict,
        target_model: str,
        reference_image=None,
        source_prompt: str = "",
        evaluation_focus: str = "balanced",
        creative_brief=None,
        character_bible=None,
        world_bible=None,
        shot_list=None,
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        require_declared_capability(ollama_model, "vision")
        context = compact_asset_context(
            creative_brief, character_bible, world_bible, shot_list
        )
        images = [tensor_to_base64(image)]
        if reference_image is not None:
            images.append(tensor_to_base64(reference_image))
        execution = run_structured_task(
            task_id="image_evaluator",
            template_version="1",
            model_profile=ollama_model,
            prompt=build_image_evaluation_prompt(
                source_prompt,
                target_model,
                evaluation_focus,
                reference_image is not None,
                context,
            ),
            system=IMAGE_EVALUATOR_SYSTEM,
            schema=build_evaluation_schema("image", target_model),
            num_predict=4096,
            temperature=min(float(ollama_model.get("temperature", 0.2)), 0.3),
            repair_once=True,
            cache_policy=cache_policy,
            images=images,
            generate_fn=generate,
        )
        value = execution.structured.value
        if not execution.structured.valid or not isinstance(value, dict):
            return _result(execution, {}, False, execution.report)
        data = normalize_evaluation(value, execution.provenance)
        return _result(execution, data, True, "Evaluation is schema-valid; score and verdict were recomputed locally.")


class OllamaEvaluationCompare:
    """Compare two typed evaluations deterministically."""

    CATEGORY = "ComfyUI-OMG/Evaluation"
    FUNCTION = "compare"
    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("winner", "score_difference", "recommendation", "comparison_json")
    DESCRIPTION = "Compare two OMG Evaluations using their locally normalized scores."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "evaluation_a": ("OMG_EVALUATION", {"forceInput": True}),
                "evaluation_b": ("OMG_EVALUATION", {"forceInput": True}),
            },
            "optional": {
                "label_a": ("STRING", {"default": "Candidate A"}),
                "label_b": ("STRING", {"default": "Candidate B"}),
                "tie_tolerance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0}),
            },
        }

    def compare(
        self,
        evaluation_a: dict,
        evaluation_b: dict,
        label_a: str = "Candidate A",
        label_b: str = "Candidate B",
        tie_tolerance: float = 0.5,
    ):
        data_a = require_evaluation(evaluation_a)
        data_b = require_evaluation(evaluation_b)
        score_a = float(data_a.get("overall_score", 0))
        score_b = float(data_b.get("overall_score", 0))
        difference = abs(score_a - score_b)
        if difference <= tie_tolerance:
            winner = "tie"
            recommendation = "Scores are effectively tied; compare issue severity and artistic intent."
        elif score_a > score_b:
            winner = label_a
            recommendation = str(data_a.get("recommendation", ""))
        else:
            winner = label_b
            recommendation = str(data_b.get("recommendation", ""))
        comparison = {
            "winner": winner,
            "score_difference": difference,
            "candidate_a": {"label": label_a, "score": score_a, "verdict": data_a.get("verdict")},
            "candidate_b": {"label": label_b, "score": score_b, "verdict": data_b.get("verdict")},
            "recommendation": recommendation,
        }
        return (winner, difference, recommendation, json.dumps(comparison, indent=2, ensure_ascii=False))


class OllamaEvaluationInspect:
    """Expose a typed evaluation through focused string and numeric outputs."""

    CATEGORY = "ComfyUI-OMG/Evaluation"
    FUNCTION = "inspect_evaluation"
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "FLOAT",
    )
    RETURN_NAMES = (
        "evaluation_type",
        "target_model",
        "overall_score",
        "verdict",
        "criteria_json",
        "strengths",
        "issues_json",
        "recommendation",
        "confidence",
    )
    DESCRIPTION = "Inspect a typed OMG Evaluation through scores, criteria, issues, and recommendation."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"evaluation": ("OMG_EVALUATION", {"forceInput": True})}}

    def inspect_evaluation(self, evaluation: dict):
        data = require_evaluation(evaluation)
        return (
            str(data.get("evaluation_type", "")),
            str(data.get("target_model", "")),
            float(data.get("overall_score", 0)),
            str(data.get("verdict", "")),
            json.dumps(data.get("criteria", {}), indent=2, ensure_ascii=False, sort_keys=True),
            "\n".join(str(item) for item in data.get("strengths", [])),
            json.dumps(data.get("issues", []), indent=2, ensure_ascii=False),
            str(data.get("recommendation", "")),
            float(data.get("confidence", 0)),
        )
