"""Fast deterministic cross-image consistency and drift diagnostics.

This module intentionally uses bounded lexical/claim comparisons only. It performs no
image decoding, embeddings, model inference, network requests, or extra Ollama calls.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

MAX_ANALYSES = 128
_MAX_TEXT_CHARS = 4_000
_MAX_TOKENS_PER_FIELD = 192
_MAX_CONFLICT_DETAILS_PER_FIELD = 12

DOMAIN_FIELDS: dict[str, tuple[str, ...]] = {
    "identity": ("subject", "body_details"),
    "clothing": ("clothing",),
    "environment": ("location",),
    "camera": ("camera",),
    "lighting": ("lighting",),
    "palette": ("color_palette",),
    "style": ("art_style",),
}

_FIELD_WEIGHTS = {
    "identity": 0.22,
    "clothing": 0.18,
    "environment": 0.17,
    "camera": 0.10,
    "lighting": 0.13,
    "palette": 0.08,
    "style": 0.12,
}

_STOPWORDS = {
    "about", "above", "across", "after", "against", "also", "among", "and", "are",
    "around", "because", "been", "being", "below", "both", "but", "camera", "clearly",
    "detail", "details", "from", "front", "have", "image", "into", "level", "more",
    "most", "not", "only", "other", "over", "same", "scene", "shown", "shows", "that",
    "the", "their", "there", "these", "they", "this", "through", "under", "very", "visible",
    "what", "when", "where", "which", "while", "with", "without",
}

_COLOR_ALIASES = {
    "grey": "gray",
    "turquoise": "teal",
    "aqua": "cyan",
    "magenta": "pink",
    "blond": "blonde",
}
_COLOR_TERMS = {
    "black", "white", "gray", "red", "orange", "amber", "yellow", "green", "olive",
    "teal", "cyan", "blue", "navy", "indigo", "violet", "purple", "pink", "brown",
    "beige", "cream", "tan", "silver", "gold", "charcoal", "blonde", "auburn",
}
_GARMENTS = {
    "shirt", "blouse", "sweater", "jacket", "coat", "hoodie", "dress", "skirt", "suit",
    "uniform", "armor", "robe", "kimono", "vest", "corset", "trousers", "pants", "jeans",
    "shorts", "leggings", "boots", "shoes", "sandals", "heels", "sneakers", "hat", "helmet",
    "gloves", "scarf", "belt", "tie", "cape", "cloak", "gown", "swimsuit", "bikini",
}
_MATERIALS = {
    "cotton", "linen", "denim", "leather", "silk", "satin", "wool", "velvet", "lace",
    "nylon", "latex", "rubber", "metal", "plastic", "canvas", "chiffon", "suede",
}
_SETTING_TERMS = {
    "studio", "street", "city", "forest", "woodland", "beach", "coast", "desert", "mountain",
    "cavern", "cave", "office", "bedroom", "kitchen", "bathroom", "warehouse", "factory",
    "laboratory", "observatory", "spacecraft", "station", "vehicle", "train", "garden", "field",
    "stage", "restaurant", "cafe", "school", "hospital", "temple", "castle", "ruins", "underwater",
}
_FRAMING_TERMS = {
    "extreme close-up", "close-up", "closeup", "medium shot", "medium-wide", "wide shot",
    "full-body", "full body", "portrait", "macro", "aerial", "establishing shot",
}
_ANGLE_TERMS = {
    "eye-level", "eye level", "high angle", "low angle", "bird's-eye", "birds-eye",
    "worm's-eye", "worms-eye", "dutch angle", "overhead", "profile", "three-quarter",
}
_LIGHT_SOURCES = {
    "sunlight", "sun", "window", "softbox", "lamp", "neon", "candle", "fire", "moonlight",
    "practical", "flash", "spotlight", "skylight", "ambient", "rim light", "key light", "fill light",
}
_STYLE_MEDIA = {
    "photograph": "photograph",
    "photography": "photograph",
    "photo": "photograph",
    "photorealistic": "photograph",
    "anime": "anime",
    "manga": "anime",
    "illustration": "illustration",
    "drawing": "drawing",
    "sketch": "drawing",
    "painting": "painting",
    "painterly": "painting",
    "watercolor": "watercolor",
    "oil painting": "oil painting",
    "3d render": "3d render",
    "digital render": "3d render",
    "cgi": "3d render",
    "pixel art": "pixel art",
    "vector": "vector",
    "claymation": "claymation",
}

# Categories here represent mutually incompatible claims when both frames make a claim.
_EXCLUSIVE_CATEGORIES = {
    "identity": {"person_count", "presentation", "age_group", "hair_color"},
    "clothing": {"coverage"},
    "environment": {"enclosure", "setting"},
    "lighting": {"time_of_day"},
    "style": {"medium"},
}
_MAJOR_CONFLICT_FIELDS = {"identity", "clothing", "environment", "lighting", "style"}


def _normalized_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.casefold()[:_MAX_TEXT_CHARS])
    normalized = []
    for word in words:
        word = _COLOR_ALIASES.get(word, word)
        if word in _STOPWORDS or (len(word) < 3 and not word.isdigit()):
            continue
        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"
        elif word.endswith("s") and len(word) > 5 and not word.endswith("ss"):
            word = word[:-1]
        normalized.append(word)
        if len(normalized) >= _MAX_TOKENS_PER_FIELD:
            break
    return normalized


def _token_set(text: str) -> set[str]:
    return set(_normalized_words(text))


def _find_phrases(text: str, phrases: set[str]) -> set[str]:
    lower = f" {text.casefold()} "
    return {phrase for phrase in phrases if re.search(rf"\b{re.escape(phrase)}\b", lower)}


def _colors(text: str) -> set[str]:
    words = {_COLOR_ALIASES.get(word, word) for word in re.findall(r"[a-z]+", text.casefold())}
    values = words & _COLOR_TERMS
    values.update(value.upper() for value in re.findall(r"#[0-9a-fA-F]{6}\b", text))
    return values


def _person_count(text: str) -> set[str]:
    lower = text.casefold()
    for word, value in (("single", "1"), ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4")):
        if re.search(rf"\b{word}\s+(?:adult|person|people|subject|character|explorer|child|man|woman)", lower):
            return {value}
    match = re.search(r"\b([1-9])\s+(?:people|persons|subjects|characters|adults|children|men|women)\b", lower)
    return {match.group(1)} if match else set()


def _hair_colors(text: str) -> set[str]:
    words = _normalized_words(text)
    found = set()
    for index, word in enumerate(words):
        if word not in {"hair", "haired"}:
            continue
        nearby = words[max(0, index - 4): index + 5]
        found.update(set(nearby) & _COLOR_TERMS)
    return found


def _coverage(text: str) -> set[str]:
    lower = text.casefold()
    if any(term in lower for term in ("completely nude", "fully nude", "naked", "unclothed")):
        return {"nude"}
    if any(term in lower for term in ("topless", "bottomless", "partially clothed", "minimally clothed")):
        return {"partial"}
    if "fully clothed" in lower:
        return {"full"}
    return set()


def _features(domain: str, text: str) -> dict[str, set[str]]:
    lower = text.casefold()
    features: dict[str, set[str]] = {}
    if domain == "identity":
        features["person_count"] = _person_count(text)
        presentation = set()
        if re.search(r"\b(?:female|woman|women|girl)\b", lower):
            presentation.add("female")
        if re.search(r"\b(?:male|man|men|boy)\b", lower):
            presentation.add("male")
        features["presentation"] = presentation
        age = set()
        for pattern, value in (
            (r"\b(?:baby|infant|toddler|child)\b", "child"),
            (r"\bteen(?:ager)?\b", "teen"),
            (r"\b(?:adult|young adult|middle-aged)\b", "adult"),
            (r"\b(?:elderly|senior)\b", "senior"),
        ):
            if re.search(pattern, lower):
                age.add(value)
        features["age_group"] = age
        features["hair_color"] = _hair_colors(text)
    elif domain == "clothing":
        features["coverage"] = _coverage(text)
        features["garment"] = _find_phrases(text, _GARMENTS)
        features["material"] = _find_phrases(text, _MATERIALS)
        features["color"] = _colors(text)
    elif domain == "environment":
        enclosure = set()
        if re.search(r"\b(?:indoor|indoors|interior|inside)\b", lower):
            enclosure.add("indoor")
        if re.search(r"\b(?:outdoor|outdoors|exterior|outside)\b", lower):
            enclosure.add("outdoor")
        features["enclosure"] = enclosure
        features["setting"] = _find_phrases(text, _SETTING_TERMS)
    elif domain == "camera":
        features["lens"] = {f"{value}mm" for value in re.findall(r"\b(\d{1,3})\s*mm\b", lower)}
        features["framing"] = _find_phrases(text, _FRAMING_TERMS)
        features["angle"] = _find_phrases(text, _ANGLE_TERMS)
        depth = set()
        for value in ("shallow", "medium", "deep"):
            if re.search(rf"\b{value}\s+(?:depth|depth of field|dof)\b", lower):
                depth.add(value)
        features["depth"] = depth
    elif domain == "lighting":
        temperature = set()
        for value in ("warm", "cool", "neutral", "mixed"):
            if re.search(rf"\b{value}\b", lower):
                temperature.add(value)
        features["temperature"] = temperature
        direction = set()
        for pattern, value in (
            (r"camera[- ]left|from the left", "left"),
            (r"camera[- ]right|from the right", "right"),
            (r"backlit|backlight|from behind", "back"),
            (r"overhead|top light|from above", "top"),
            (r"front lit|frontal light", "front"),
        ):
            if re.search(pattern, lower):
                direction.add(value)
        features["direction"] = direction
        features["source"] = _find_phrases(text, _LIGHT_SOURCES)
        time = set()
        for pattern, value in (
            (r"\b(?:night|nighttime|moonlit)\b", "night"),
            (r"\b(?:day|daytime|daylight)\b", "day"),
            (r"\b(?:sunrise|dawn)\b", "dawn"),
            (r"\b(?:sunset|dusk|golden hour)\b", "dusk"),
        ):
            if re.search(pattern, lower):
                time.add(value)
        features["time_of_day"] = time
    elif domain == "palette":
        features["color"] = _colors(text)
    elif domain == "style":
        media = set()
        for phrase, canonical in _STYLE_MEDIA.items():
            if re.search(rf"\b{re.escape(phrase)}\b", lower):
                media.add(canonical)
        features["medium"] = media
        finish = set()
        for value in ("cinematic", "editorial", "commercial", "documentary", "realistic", "stylized"):
            if re.search(rf"\b{value}\b", lower):
                finish.add(value)
        features["finish"] = finish
    return {key: values for key, values in features.items() if values}


def _dice(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2.0 * len(left & right) / (len(left) + len(right))


def _feature_similarity(left: dict[str, set[str]], right: dict[str, set[str]]) -> float | None:
    scores = []
    for category in left.keys() & right.keys():
        scores.append(_dice(left[category], right[category]))
    return sum(scores) / len(scores) if scores else None


def _joined_text(analysis: dict[str, Any], fields: tuple[str, ...]) -> str:
    return " ".join(str(analysis.get(field, "") or "").strip() for field in fields).strip()


def _field_report(domain: str, prepared: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    useful = [item for item in prepared if item["text"]]
    if len(useful) < 2:
        return (
            {
                "classification": "insufficient_evidence",
                "drift_score": 0.0,
                "mean_similarity": 0.0,
                "evidence_count": len(useful),
                "comparison_count": 0,
                "stable_terms": [],
                "variable_terms": [],
                "conflict_count": 0,
            },
            [],
        )

    similarities = []
    conflicts = []
    conflict_count = 0
    for left_index, left in enumerate(useful):
        for right in useful[left_index + 1:]:
            lexical = _dice(left["tokens"], right["tokens"])
            feature = _feature_similarity(left["features"], right["features"])
            similarity = lexical if feature is None else (0.4 * lexical + 0.6 * feature)
            pair_conflicts = []
            for category in _EXCLUSIVE_CATEGORIES.get(domain, set()):
                left_values = left["features"].get(category, set())
                right_values = right["features"].get(category, set())
                if left_values and right_values and left_values.isdisjoint(right_values):
                    pair_conflicts.append(
                        {
                            "field": domain,
                            "category": category,
                            "severity": "major" if domain in _MAJOR_CONFLICT_FIELDS else "minor",
                            "sources": [left["label"], right["label"]],
                            "values": [sorted(left_values), sorted(right_values)],
                            "message": (
                                f"{left['label']} reports {category}={','.join(sorted(left_values))}; "
                                f"{right['label']} reports {category}={','.join(sorted(right_values))}"
                            ),
                        }
                    )
            if pair_conflicts:
                similarity = min(similarity, 0.25)
                conflict_count += len(pair_conflicts)
                remaining = _MAX_CONFLICT_DETAILS_PER_FIELD - len(conflicts)
                if remaining > 0:
                    conflicts.extend(pair_conflicts[:remaining])
            similarities.append(similarity)

    mean_similarity = sum(similarities) / len(similarities)
    drift = max(0.0, min(1.0, 1.0 - mean_similarity))
    if conflicts:
        drift = max(drift, 0.65)
        classification = "conflicting"
    elif drift <= 0.45:
        classification = "stable"
    else:
        classification = "variable"

    token_frequency = Counter(token for item in useful for token in item["tokens"])
    consensus_minimum = max(2, math.ceil(len(useful) * 0.65))
    stable_terms = sorted(
        (token for token, count in token_frequency.items() if count >= consensus_minimum),
        key=lambda token: (-token_frequency[token], token),
    )[:16]
    variable_terms = sorted(
        (token for token, count in token_frequency.items() if count < consensus_minimum),
        key=lambda token: (-token_frequency[token], token),
    )[:16]
    return (
        {
            "classification": classification,
            "drift_score": round(drift, 3),
            "mean_similarity": round(mean_similarity, 3),
            "evidence_count": len(useful),
            "comparison_count": len(similarities),
            "stable_terms": stable_terms,
            "variable_terms": variable_terms,
            "conflict_count": conflict_count,
        },
        conflicts,
    )


def build_consistency_report(
    analyses: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Compare 2-128 structured image analyses without model or image work."""

    if len(analyses) > MAX_ANALYSES:
        raise ValueError(f"At most {MAX_ANALYSES} image analyses can be compared at once")
    normalized = [
        (str(label or f"analysis_{index}"), analysis)
        for index, (label, analysis) in enumerate(analyses, start=1)
        if isinstance(analysis, dict)
    ]
    fields = {}
    all_conflicts = []
    for domain, source_fields in DOMAIN_FIELDS.items():
        prepared = []
        for label, analysis in normalized:
            text = _joined_text(analysis, source_fields)
            prepared.append(
                {
                    "label": label,
                    "text": text,
                    "tokens": _token_set(text),
                    "features": _features(domain, text),
                }
            )
        field_report, conflicts = _field_report(domain, prepared)
        fields[domain] = field_report
        all_conflicts.extend(conflicts)

    weighted_drift = 0.0
    weight_total = 0.0
    for domain, report in fields.items():
        if report["classification"] == "insufficient_evidence":
            continue
        weight = _FIELD_WEIGHTS[domain]
        weighted_drift += report["drift_score"] * weight
        weight_total += weight
    overall_drift = round(weighted_drift / weight_total, 3) if weight_total else 0.0
    overall_consistency = round(1.0 - overall_drift, 3) if weight_total else 0.0
    major_conflicts = sum(
        report["conflict_count"]
        for domain, report in fields.items()
        if domain in _MAJOR_CONFLICT_FIELDS
    )
    if major_conflicts:
        status = "conflicting"
        overall_consistency = min(overall_consistency, 0.49)
        overall_drift = max(overall_drift, 0.51)
    elif not weight_total or len(normalized) < 2:
        status = "insufficient_evidence"
    elif overall_drift <= 0.45:
        status = "stable"
    else:
        status = "variable"
    grade = "high" if overall_consistency >= 0.72 else (
        "medium" if overall_consistency >= 0.48 else "low"
    )
    warnings = []
    for domain, report in fields.items():
        if report["classification"] == "variable":
            warnings.append(
                {
                    "severity": "minor",
                    "field": domain,
                    "message": f"{domain} varies across analyses (drift {report['drift_score']:.3f})",
                }
            )
    if major_conflicts:
        warnings.append(
            {
                "severity": "major",
                "field": "cross_image",
                "message": f"{major_conflicts} incompatible cross-image claim(s) require review",
            }
        )
    return {
        "schema": "omg.multi_image_consistency",
        "version": 1,
        "analysis_count": len(normalized),
        "overall_consistency": overall_consistency,
        "overall_drift": overall_drift,
        "grade": grade,
        "status": status,
        "fields": fields,
        "conflicts": all_conflicts,
        "warnings": warnings,
        "limits": {
            "max_analyses": MAX_ANALYSES,
            "max_text_chars_per_field": _MAX_TEXT_CHARS,
            "max_conflict_details_per_field": _MAX_CONFLICT_DETAILS_PER_FIELD,
        },
        "method": (
            "bounded deterministic lexical and explicit-claim comparison; "
            "no image decoding, embeddings, network requests, or model calls"
        ),
    }


def compact_consistency_guidance(report: dict[str, Any], max_chars: int = 1_600) -> str:
    """Build bounded synthesis guidance from a consistency report."""

    if report.get("analysis_count", 0) < 2:
        return ""
    lines = [
        "Deterministic cross-image check (use as guardrails, not as new visual evidence):",
        f"status={report.get('status', '')}; consistency={report.get('overall_consistency', 0):.3f}",
    ]
    for domain, field in report.get("fields", {}).items():
        classification = field.get("classification", "")
        stable = ", ".join(field.get("stable_terms", [])[:8])
        line = f"- {domain}: {classification}, drift={field.get('drift_score', 0):.3f}"
        if stable:
            line += f"; repeated terms={stable}"
        lines.append(line)
    for conflict in report.get("conflicts", [])[:8]:
        lines.append(f"- REVIEW {conflict.get('field')}: {conflict.get('message', '')}")
    lines.append(
        "Preserve repeated evidence; describe true keyframe changes as transitions; "
        "do not silently choose between incompatible identity, wardrobe, setting, lighting, or style claims."
    )
    return "\n".join(lines)[:max_chars]
