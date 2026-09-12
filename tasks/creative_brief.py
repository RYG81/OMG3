"""Creative Brief schema, prompt template, and deterministic target adapters."""

from __future__ import annotations

from typing import Any

CREATIVE_BRIEF_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "intent": {"type": "string"},
        "subject": {"type": "string"},
        "action": {"type": "string"},
        "environment": {"type": "string"},
        "composition": {"type": "string"},
        "camera": {
            "type": "object",
            "properties": {
                "shot": {"type": "string"},
                "angle": {"type": "string"},
                "lens": {"type": "string"},
                "movement": {"type": "string"},
            },
            "required": ["shot", "angle", "lens", "movement"],
            "additionalProperties": False,
        },
        "lighting": {"type": "string"},
        "color_palette": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "mood": {"type": "string"},
        "visual_style": {"type": "string"},
        "materials_textures": {"type": "string"},
        "continuity_constraints": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
        },
        "must_include": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "must_avoid": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "negative_prompt": {"type": "string"},
        "aspect_ratio": {"type": "string"},
        "target_model": {"type": "string"},
    },
    "required": [
        "title",
        "intent",
        "subject",
        "action",
        "environment",
        "composition",
        "camera",
        "lighting",
        "color_palette",
        "mood",
        "visual_style",
        "materials_textures",
        "continuity_constraints",
        "must_include",
        "must_avoid",
        "negative_prompt",
        "aspect_ratio",
        "target_model",
    ],
    "additionalProperties": False,
}

CREATIVE_BRIEF_SYSTEM = """You are a senior visual creative director.
Convert the user's concept and constraints into one internally consistent production brief.
Use concrete, visible details rather than vague quality words. Do not introduce named people,
brands, copyrighted characters, or identity-sensitive details unless the user explicitly supplied
them. Keep must-include, must-avoid, and continuity requirements distinct. Camera movement should
be 'static' for a still image target. Return only data that satisfies the supplied JSON Schema."""

TARGET_MODELS = ["General", "Flux", "SDXL", "Qwen Image", "Wan Video", "LTX Video"]


def build_creative_brief_prompt(
    concept: str,
    target_model: str,
    aspect_ratio: str,
    style_direction: str,
    constraints: str,
    detail_level: str,
) -> str:
    return "\n\n".join(
        (
            f"Concept:\n{concept.strip()}",
            f"Target generation model/family: {target_model}",
            f"Aspect ratio: {aspect_ratio}",
            f"Desired detail level: {detail_level}",
            f"Style direction:\n{style_direction.strip() or 'Infer an appropriate coherent style.'}",
            f"Production constraints:\n{constraints.strip() or 'No additional constraints.'}",
            "Create one production-ready creative brief. Return only the schema-compliant JSON.",
        )
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _list_text(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def compile_creative_brief_prompt(
    brief: dict[str, Any], target_model: str | None = None, extra_instructions: str = ""
) -> tuple[str, str]:
    """Compile one validated brief into a deterministic target-family prompt pair."""

    target = target_model or _text(brief.get("target_model")) or "General"
    camera = brief.get("camera", {}) if isinstance(brief.get("camera"), dict) else {}
    palette = ", ".join(_list_text(brief.get("color_palette")))
    must_include = ", ".join(_list_text(brief.get("must_include")))
    continuity = ", ".join(_list_text(brief.get("continuity_constraints")))
    base_parts = [
        _text(brief.get("subject")),
        _text(brief.get("action")),
        _text(brief.get("environment")),
        _text(brief.get("composition")),
        " ".join(
            part
            for part in (
                _text(camera.get("shot")),
                _text(camera.get("angle")),
                _text(camera.get("lens")),
                _text(camera.get("movement")),
            )
            if part
        ),
        _text(brief.get("lighting")),
        f"color palette: {palette}" if palette else "",
        _text(brief.get("mood")),
        _text(brief.get("visual_style")),
        _text(brief.get("materials_textures")),
        f"must include: {must_include}" if must_include else "",
        f"continuity: {continuity}" if continuity else "",
        f"aspect ratio: {_text(brief.get('aspect_ratio'))}" if brief.get("aspect_ratio") else "",
        _text(extra_instructions),
    ]
    parts = [part for part in base_parts if part]

    if target == "SDXL":
        positive = ", ".join(parts)
    elif target in {"Wan Video", "LTX Video"}:
        positive = ". ".join(part.rstrip(". ") for part in parts) + "."
        positive += " Maintain temporal consistency and physically coherent motion."
    else:
        positive = ". ".join(part.rstrip(". ") for part in parts) + "."

    negative_parts = [_text(brief.get("negative_prompt")), *_list_text(brief.get("must_avoid"))]
    negative = ", ".join(dict.fromkeys(part for part in negative_parts if part))
    return positive.strip(), negative.strip()
