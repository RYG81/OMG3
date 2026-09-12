"""Typed prompt/image evaluation schema, context compaction, and score normalization."""

from __future__ import annotations

import json
from typing import Any

EVALUATION_TARGETS = [
    "General",
    "Flux",
    "SDXL",
    "Qwen Image",
    "Wan Video",
    "LTX Video",
]
EVALUATION_FOCUS = [
    "balanced",
    "prompt quality",
    "intent adherence",
    "continuity",
    "composition",
    "model compatibility",
]
CRITERIA = (
    "intent_adherence",
    "subject_consistency",
    "composition",
    "lighting",
    "style",
    "continuity",
    "technical_quality",
)
_WEIGHTS = {
    "intent_adherence": 0.25,
    "subject_consistency": 0.15,
    "composition": 0.15,
    "lighting": 0.10,
    "style": 0.10,
    "continuity": 0.15,
    "technical_quality": 0.10,
}


def build_evaluation_schema(evaluation_type: str, target_model: str) -> dict[str, Any]:
    criterion = {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 100},
            "feedback": {"type": "string"},
        },
        "required": ["score", "feedback"],
        "additionalProperties": False,
    }
    issue = {
        "type": "object",
        "properties": {
            "severity": {"enum": ["critical", "major", "minor"]},
            "category": {"type": "string"},
            "description": {"type": "string"},
            "fix": {"type": "string"},
        },
        "required": ["severity", "category", "description", "fix"],
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {
        "evaluation_type": {"const": evaluation_type},
        "target_model": {"const": target_model},
        "overall_score": {"type": "number", "minimum": 0, "maximum": 100},
        "verdict": {"enum": ["pass", "needs_revision", "fail"]},
        "criteria": {
            "type": "object",
            "properties": {name: criterion for name in CRITERIA},
            "required": list(CRITERIA),
            "additionalProperties": False,
        },
        "strengths": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "issues": {"type": "array", "items": issue, "maxItems": 20},
        "missing_requirements": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
        },
        "recommendation": {"type": "string"},
        "improved_prompt": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


PROMPT_EVALUATOR_SYSTEM = """You are a rigorous AI-generation prompt evaluator. Score only against
the user's intent, supplied typed assets, internal consistency, visual specificity, composition,
lighting, style coherence, continuity, and target-model compatibility. Do not reward verbosity by
itself. Identify concrete issues and actionable fixes. The improved prompt must preserve intent and
all immutable asset rules. Return only schema-compliant JSON."""

IMAGE_EVALUATOR_SYSTEM = """You are a rigorous visual output evaluator. Image 1 is the candidate.
If Image 2 exists it is the reference. Evaluate visible evidence against the source prompt and typed
assets: intent, identity, environment, composition, lighting, style, continuity, anatomy/artifacts,
and target-model goals. Do not invent hidden details. Give actionable fixes and an improved prompt.
Return only schema-compliant JSON."""


def _asset_data(envelope: Any, schema_name: str) -> dict[str, Any] | None:
    if envelope is None:
        return None
    if not isinstance(envelope, dict) or envelope.get("schema") != schema_name:
        raise ValueError(f"Expected {schema_name} envelope")
    if envelope.get("valid") is False:
        raise ValueError(f"{schema_name} is marked invalid")
    data = envelope.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{schema_name} has no data")
    return data


def compact_asset_context(
    creative_brief=None,
    character_bible=None,
    world_bible=None,
    shot_list=None,
) -> str:
    payload: dict[str, Any] = {}
    creative = _asset_data(creative_brief, "omg.creative_brief")
    if creative:
        payload["creative_brief"] = creative
    character = _asset_data(character_bible, "omg.character_bible")
    if character:
        payload["character_bible"] = {
            key: character.get(key)
            for key in (
                "character_id",
                "name",
                "core_identity",
                "face",
                "hair",
                "eyes",
                "body_proportions",
                "distinguishing_features",
                "default_wardrobe",
                "style_anchors",
                "immutable_anchors",
                "master_prompt",
            )
        }
    world = _asset_data(world_bible, "omg.world_bible")
    if world:
        payload["world_bible"] = {
            key: world.get(key)
            for key in (
                "world_id",
                "premise",
                "era_technology",
                "geography",
                "architecture",
                "color_palette",
                "lighting_rules",
                "visual_style",
                "immutable_world_rules",
                "forbidden_drift",
                "master_environment_prompt",
            )
        }
        payload["world_bible"]["key_location_names"] = [
            item.get("name", "")
            for item in world.get("key_locations", [])
            if isinstance(item, dict)
        ]
    shots = _asset_data(shot_list, "omg.shot_list")
    if shots:
        payload["shot_list"] = {
            "shot_list_id": shots.get("shot_list_id"),
            "sequence_summary": shots.get("sequence_summary"),
            "target_model": shots.get("target_model"),
            "aspect_ratio": shots.get("aspect_ratio"),
            "continuity": shots.get("continuity"),
            "shots": [
                {
                    "shot_id": item.get("shot_id"),
                    "purpose": item.get("purpose"),
                    "video_prompt": item.get("video_prompt"),
                    "continuity_locks": item.get("continuity_locks"),
                }
                for item in shots.get("shots", [])
                if isinstance(item, dict)
            ],
        }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if len(text) > 40000:
        text = text[:40000] + "\n... [typed asset context truncated at 40,000 characters]"
    return text


def build_prompt_evaluation_prompt(
    prompt: str,
    target_model: str,
    intended_result: str,
    focus: str,
    asset_context: str,
) -> str:
    return "\n\n".join(
        (
            f"Target model/family: {target_model}",
            f"Evaluation focus: {focus}",
            f"Intended result:\n{intended_result.strip() or 'Infer from the prompt and assets.'}",
            f"Prompt to evaluate:\n{prompt.strip()}",
            f"Typed asset requirements:\n{asset_context or 'None supplied.'}",
            "Evaluate the prompt and propose a corrected prompt where useful.",
        )
    )


def build_image_evaluation_prompt(
    source_prompt: str,
    target_model: str,
    focus: str,
    has_reference: bool,
    asset_context: str,
) -> str:
    image_note = (
        "Image 1 is the candidate output and Image 2 is the reference image."
        if has_reference
        else "Image 1 is the candidate output; no separate reference image was supplied."
    )
    return "\n\n".join(
        (
            image_note,
            f"Target model/family: {target_model}",
            f"Evaluation focus: {focus}",
            f"Source prompt:\n{source_prompt.strip() or 'No source prompt supplied.'}",
            f"Typed asset requirements:\n{asset_context or 'None supplied.'}",
            "Evaluate visible adherence and provide a better prompt for the next generation.",
        )
    )


def normalize_evaluation(data: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    criteria = data.get("criteria", {})
    weighted = sum(float(criteria[name]["score"]) * _WEIGHTS[name] for name in CRITERIA)
    issues = data.get("issues", [])
    has_critical = any(
        isinstance(issue, dict) and issue.get("severity") == "critical" for issue in issues
    )
    if has_critical or weighted < 50:
        derived_verdict = "fail"
    elif weighted < 75:
        derived_verdict = "needs_revision"
    else:
        derived_verdict = "pass"
    result = dict(data)
    result["evaluation_id"] = str(provenance.get("request_fingerprint", ""))[:20]
    result["model_overall_score"] = data.get("overall_score")
    result["model_verdict"] = data.get("verdict")
    result["overall_score"] = round(weighted, 2)
    result["verdict"] = derived_verdict
    result["score_weights"] = dict(_WEIGHTS)
    return result


def make_evaluation_envelope(data: dict[str, Any], provenance: dict[str, Any], valid: bool) -> dict[str, Any]:
    return {
        "schema": "omg.evaluation",
        "version": 1,
        "valid": valid,
        "data": data if valid else {},
        "provenance": provenance,
    }


def require_evaluation(envelope: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(envelope, dict) or envelope.get("schema") != "omg.evaluation":
        raise ValueError("Input is not an OMG Evaluation envelope")
    if envelope.get("valid") is False:
        raise ValueError("Evaluation is marked invalid")
    data = envelope.get("data")
    if not isinstance(data, dict) or not data:
        raise ValueError("Evaluation contains no validated data")
    return data
