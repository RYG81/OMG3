"""Domain-level quality checks applied after JSON Schema validation."""

from __future__ import annotations

import re
from typing import Any

from .image_evidence import evidence_quality_errors

QUALITY_RULES_VERSION = "4"

# Minimum useful lengths for fields that are promised as directly usable outputs.
_MIN_LENGTHS: dict[str, dict[str, int]] = {
    "prompt_enhancer": {"enhanced_prompt": 40},
    "scene_director": {"scene_prompt": 80, "composition_guide": 30, "lighting_setup": 30},
    "character_sheet": {"character_summary": 80, "master_sheet": 100},
    "image_analyzer": {"reconstruction_prompt": 80},
    "image_sequence_frame": {"reconstruction_prompt": 80},
    "photoset_image_analysis": {"reconstruction_prompt": 80},
    "web_image_analysis": {"reconstruction_prompt": 80},
    "image_sequence_video_prompt": {"video_prompt": 80},
    "video_shot_director": {"sequence_prompt": 80, "shot_list": 40},
    "prompt_critic": {"improved_prompt": 40, "improvement_suggestions": 20},
    "negative_prompt": {"negative_prompt": 20},
    "advanced_scene_director": {"scene_prompt": 80},
    "character_anchor": {"compact_anchor_prompt": 40},
    "outfit_sheet": {"outfit_sheet_prompt": 80},
    "pose_sheet": {"pose_sheet_prompt": 80},
    "expression_sheet": {"expression_sheet_prompt": 80},
    "continuity_check": {"fixed_master_prompt": 50},
    "photoset_incremental_update": {"consistent_prompt": 80},
    "photoset_final_synthesis": {"consistent_prompt": 80},
    "auto_tagger": {"booru_tags": 15},
    "style_identifier": {"replication_prompt": 40},
    "regional_prompts": {"full_composition": 40},
    "image_comparator": {"comparison_summary": 40},
    "inpaint_prompt": {"inpaint_prompt": 30, "context_description": 30},
    "style_transfer": {"styled_prompt": 60, "style_description": 40},
    "environment_transform": {"transformed_prompt": 60},
    "image_merger": {"merged_prompt": 80},
    "image_edit_prompt": {"edit_prompt": 60},
    "image_to_story": {"story": 100},
    "pose_descriptor": {"pose_description": 40},
    "outfit_generator": {"outfit_prompt": 60},
    "subject_builder": {"full_description": 80},
    "creature_creator": {"creature_prompt": 80},
    "hand_pose_helper": {"hand_prompt": 50},
    "lighting_designer": {"lighting_prompt": 50},
    "texture_material": {"material_prompt": 50},
    "action_choreographer": {"action_prompt": 80},
    "background_generator": {"background_prompt": 80},
    "emotion_director": {"emotional_prompt": 60},
    "detail_injector": {"enhanced_prompt": 60},
    "controlnet_helper": {"optimized_prompt": 60},
    "prompt_combiner": {"combined_prompt": 60},
    "prompt_translator": {"translated_prompt": 40},
    "aspect_optimizer": {"optimized_prompt": 60},
    "creative_brief": {"subject": 20, "environment": 20, "composition": 15},
    "character_bible_create": {"core_identity": 40, "master_prompt": 80},
    "character_bible_update": {"core_identity": 40, "master_prompt": 80},
    "world_bible_create": {"premise": 40, "master_environment_prompt": 80},
    "world_bible_update": {"premise": 40, "master_environment_prompt": 80},
}

_PROMPT_FIELDS = {
    field
    for rules in _MIN_LENGTHS.values()
    for field in rules
    if "prompt" in field or field in {"scene_prompt", "story", "full_description"}
}
_PROMPT_FIELDS.update({"positive_prompt", "video_prompt", "master_sheet"})
_LABEL_PREFIX = re.compile(
    r"^\s*(?:positive prompt|negative prompt|scene prompt|enhanced prompt|output|result)\s*:",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(r"\[(?:insert|describe|subject|object|details?|placeholder)[^\]]*\]", re.I)
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
_PRIMARY_FIELDS = {
    "prompt_enhancer": "enhanced_prompt",
    "prompt_translator": "translated_prompt",
    "style_transfer": "styled_prompt",
    "image_edit_prompt": "edit_prompt",
    "scene_director": "scene_prompt",
    "advanced_scene_director": "scene_prompt",
    "prompt_combiner": "combined_prompt",
    "video_shot_director": "sequence_prompt",
    "creative_brief": "subject",
}


def _rules_for(task_id: str) -> dict[str, int]:
    if task_id.startswith("video_prompt_"):
        return {"positive_prompt": 80, "shot_plan": 30, "motion_notes": 30}
    return _MIN_LENGTHS.get(task_id, {})


def _check_direct_output(field: str, text: str) -> list[str]:
    errors = []
    stripped = text.strip()
    if "```" in stripped:
        errors.append(f"{field} contains a Markdown code fence")
    if _LABEL_PREFIX.search(stripped):
        errors.append(f"{field} starts with an output label instead of usable content")
    if _PLACEHOLDER.search(stripped):
        errors.append(f"{field} contains an unresolved placeholder")
    if stripped.startswith("{") and stripped.endswith("}"):
        errors.append(f"{field} contains serialized JSON instead of a direct output")
    return errors


_STOPWORDS = {
    "about", "after", "again", "against", "being", "could", "from", "have", "into",
    "more", "only", "other", "should", "their", "there", "these", "this", "through",
    "under", "using", "very", "what", "when", "where", "which", "while", "with", "would",
    "prompt", "image", "create", "generate", "output", "target", "model",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.casefold())
    result = set()
    for word in words:
        if len(word) < 4 or word in _STOPWORDS:
            continue
        # Light suffix folding catches storm/stormy and lights/lighting without external NLP.
        for suffix in ("ing", "ies", "ed", "ly", "s", "y"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        result.add(word)
    return result


def _intent_overlap(source: str, output: str) -> bool:
    source_tokens = _tokens(source)
    if not source_tokens:
        return True
    output_tokens = _tokens(output)
    required = 1 if len(source_tokens) < 8 else 2
    return len(source_tokens & output_tokens) >= required


def _target_prompt_errors(prompt: str, target_model: str) -> list[str]:
    target = target_model.casefold()
    text = prompt.strip()
    errors = []
    comma_count = text.count(",")
    sentence_count = sum(text.count(mark) for mark in (".", "!", "?"))
    if any(name in target for name in ("flux", "qwen", "general")):
        if comma_count > 12 and sentence_count == 0:
            errors.append(f"{target_model} prompt looks like tag soup instead of natural language")
    if "sdxl" in target and "--ar" in text:
        errors.append("SDXL prompt contains Midjourney --ar syntax")
    if any(name in target for name in ("wan", "ltx", "video")):
        motion_terms = {"move", "motion", "walk", "run", "track", "pan", "dolly", "orbit", "camera", "frame"}
        if not (_tokens(text) & motion_terms):
            errors.append(f"{target_model} prompt contains no explicit camera or subject motion")
    return errors


def _shot_list_errors(value: dict[str, Any], context: dict[str, Any]) -> list[str]:
    shots = value.get("shots")
    if not isinstance(shots, list):
        return ["shots must be a list"]
    errors = []
    expected_count = context.get("expected_shot_count")
    if expected_count is not None and len(shots) != int(expected_count):
        errors.append(f"expected {expected_count} shots, received {len(shots)}")
    ids = [str(shot.get("shot_id", "")) for shot in shots if isinstance(shot, dict)]
    if len(ids) != len(set(ids)):
        errors.append("shot_id values must be unique")
    orders = [shot.get("order") for shot in shots if isinstance(shot, dict)]
    if orders != list(range(1, len(shots) + 1)):
        errors.append("shot order must be sequential from 1")
    durations = [float(shot.get("duration_seconds", 0)) for shot in shots if isinstance(shot, dict)]
    total = float(value.get("total_duration_seconds", 0))
    if abs(sum(durations) - total) > 0.05:
        errors.append("shot durations do not sum to total_duration_seconds")
    expected_total = context.get("expected_total_duration")
    if expected_total is not None and abs(total - float(expected_total)) > 0.05:
        errors.append("total_duration_seconds changed from the requested value")
    for field, context_key in (
        ("target_model", "expected_target_model"),
        ("aspect_ratio", "expected_aspect_ratio"),
        ("fps", "expected_fps"),
    ):
        expected = context.get(context_key)
        if expected is not None and value.get(field) != expected:
            errors.append(f"{field} changed from the requested value")
    continuity = value.get("continuity", {})
    if isinstance(continuity, dict):
        if continuity.get("character_ids") != context.get("expected_character_ids", []):
            errors.append("linked character_ids changed")
        if continuity.get("world_id") != context.get("expected_world_id", ""):
            errors.append("linked world_id changed")
    else:
        errors.append("continuity must be an object")
    linked_assets = bool(context.get("expected_character_ids") or context.get("expected_world_id"))
    target = str(value.get("target_model", ""))
    for index, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            errors.append(f"shot {index} is not an object")
            continue
        for field, minimum in (
            ("first_frame_prompt", 40),
            ("video_prompt", 80),
            ("last_frame_prompt", 40),
        ):
            text = str(shot.get(field, "") or "")
            if len(text.strip()) < minimum:
                errors.append(f"shot {index} {field} is too short")
            errors.extend(_check_direct_output(f"shot {index} {field}", text))
        errors.extend(_target_prompt_errors(str(shot.get("video_prompt", "")), target))
        locks = shot.get("continuity_locks", [])
        if linked_assets and (not isinstance(locks, list) or not locks):
            errors.append(f"shot {index} has no continuity locks for linked assets")
    mode = str(context.get("revision_mode", ""))
    original_shots = context.get("original_shots")
    if mode == "append_shots" and isinstance(original_shots, list):
        if shots[: len(original_shots)] != original_shots:
            errors.append("append_shots changed one or more existing shots")
        if len(shots) <= len(original_shots):
            errors.append("append_shots did not add a shot")
    elif context.get("preserve_shot_ids") and isinstance(original_shots, list):
        original_ids = [shot.get("shot_id") for shot in original_shots if isinstance(shot, dict)]
        if ids != original_ids:
            errors.append("existing shot IDs changed")
    return errors


def validate_task_output(
    task_id: str,
    value: Any,
    request_prompt: str = "",
    system_prompt: str = "",
    context: dict[str, Any] | None = None,
) -> list[str]:
    """Return actionable quality errors for a schema-valid task result."""

    if not isinstance(value, dict):
        return []
    context = context or {}
    errors = []
    for field, minimum in _rules_for(task_id).items():
        text = str(value.get(field, "") or "").strip()
        if len(text) < minimum:
            errors.append(f"{field} is too short ({len(text)} chars; minimum {minimum})")
        if field in _PROMPT_FIELDS:
            errors.extend(_check_direct_output(field, text))

    source_text = str(context.get("source_text", "") or "")
    primary_field = str(context.get("primary_field") or _PRIMARY_FIELDS.get(task_id, ""))
    primary_output = str(value.get(primary_field, "") or "") if primary_field else ""
    if source_text and primary_output and not _intent_overlap(source_text, primary_output):
        errors.append(f"{primary_field} does not preserve enough concrete concepts from the source")
    target_model = str(context.get("target_model", "") or "")
    if target_model and primary_output:
        errors.extend(_target_prompt_errors(primary_output, target_model))

    if task_id in {"image_analyzer", "image_sequence_frame", "photoset_image_analysis", "web_image_analysis"}:
        reconstruction = str(value.get("reconstruction_prompt", "") or "")
        evidence = " ".join(
            str(value.get(field, "") or "")
            for field in ("subject", "location", "lighting", "camera")
        )
        if reconstruction and evidence and not _intent_overlap(evidence, reconstruction):
            errors.append("reconstruction_prompt does not reflect enough analyzed visual evidence")
        errors.extend(evidence_quality_errors(value))
    elif task_id == "creative_brief":
        must_avoid = value.get("must_avoid", [])
        negative = str(value.get("negative_prompt", "") or "")
        avoid_tokens = [_tokens(str(item)) for item in must_avoid] if isinstance(must_avoid, list) else []
        represented = sum(bool(tokens & _tokens(negative)) for tokens in avoid_tokens if tokens)
        required = max(1, (len(avoid_tokens) + 1) // 2) if avoid_tokens else 0
        if represented < required:
            errors.append("negative_prompt does not cover enough must_avoid requirements")
        expected_target = str(context.get("target_model", "") or "")
        expected_aspect = str(context.get("aspect_ratio", "") or "")
        if expected_target and str(value.get("target_model", "")) != expected_target:
            errors.append("creative brief target_model changed from the requested value")
        if expected_aspect and str(value.get("aspect_ratio", "")) != expected_aspect:
            errors.append("creative brief aspect_ratio changed from the requested value")
    elif task_id in {"character_bible_create", "character_bible_update"}:
        anchors = value.get("immutable_anchors", [])
        master_tokens = _tokens(str(value.get("master_prompt", "") or ""))
        anchor_sets = [_tokens(str(item)) for item in anchors] if isinstance(anchors, list) else []
        represented = sum(bool(tokens & master_tokens) for tokens in anchor_sets if tokens)
        required = max(1, (len(anchor_sets) + 1) // 2) if anchor_sets else 0
        if represented < required:
            errors.append("master_prompt does not represent enough immutable identity anchors")
    elif task_id in {"world_bible_create", "world_bible_update"}:
        rules = value.get("immutable_world_rules", [])
        master_tokens = _tokens(str(value.get("master_environment_prompt", "") or ""))
        rule_sets = [_tokens(str(item)) for item in rules] if isinstance(rules, list) else []
        represented = sum(bool(tokens & master_tokens) for tokens in rule_sets if tokens)
        if rule_sets and represented == 0:
            errors.append("master_environment_prompt does not represent any immutable world rule")

    if task_id in {"shot_list_create", "shot_list_update"}:
        errors.extend(_shot_list_errors(value, context))
    elif task_id == "prompt_variations":
        variations = [
            str(value.get(f"variation_{index}", "") or "").strip()
            for index in range(1, 11)
            if value.get(f"variation_{index}")
        ]
        if any(len(item) < 25 for item in variations):
            errors.append("one or more generated variations are too short to be useful")
        if len({item.casefold() for item in variations}) != len(variations):
            errors.append("generated variations contain duplicates")
    elif task_id == "storyboard_generator":
        panels = [
            str(value.get(f"panel_{index}", "") or "").strip()
            for index in range(1, 13)
            if value.get(f"panel_{index}")
        ]
        if any(len(item) < 40 for item in panels):
            errors.append("one or more storyboard panel prompts are too short")
        if len({item.casefold() for item in panels}) != len(panels):
            errors.append("storyboard panel prompts contain duplicates")
    elif task_id == "wildcard_generator":
        prompt = str(value.get("wildcard_prompt", ""))
        if not ("{" in prompt and "|" in prompt and "}" in prompt):
            errors.append("wildcard_prompt does not contain {option1|option2} syntax")
    elif task_id == "ltx_ingredients_prompt":
        prompt = str(value.get("prompt", ""))
        for marker in ("Reference sheet:", "Generated video:"):
            if marker not in prompt:
                errors.append(f"prompt is missing required marker {marker!r}")
    elif task_id == "color_palette":
        hex_value = str(value.get("palette_hex", ""))
        if not _HEX.search(hex_value):
            errors.append("palette_hex contains no valid #RRGGBB colors")
    elif task_id in {"character_bible_create", "character_bible_update"}:
        anchors = value.get("immutable_anchors", [])
        if isinstance(anchors, list) and len({str(item).strip().casefold() for item in anchors}) != len(anchors):
            errors.append("immutable_anchors contains duplicates")
    elif task_id in {"world_bible_create", "world_bible_update"}:
        locations = value.get("key_locations", [])
        names = [str(item.get("name", "")).strip().casefold() for item in locations if isinstance(item, dict)]
        if len(names) != len(set(names)):
            errors.append("key_locations contains duplicate names")
    return errors
