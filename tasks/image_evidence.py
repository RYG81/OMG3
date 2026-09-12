"""Deterministic evidence/confidence diagnostics for structured image analysis."""

from __future__ import annotations

import re
from typing import Any

_ANALYSIS_FIELDS = (
    "subject",
    "body_details",
    "clothing",
    "pose",
    "location",
    "lighting",
    "camera",
    "color_palette",
    "art_style",
    "mood",
    "technical",
    "reconstruction_prompt",
)
_LIMITATIONS = (
    "not visible",
    "not shown",
    "cropped",
    "occluded",
    "obscured",
    "unreadable",
    "cannot be confirmed",
    "too blurred",
)
_HEDGES = (
    "maybe",
    "possibly",
    "probably",
    "appears to",
    "seems to",
    "could be",
    "might be",
    "and/or",
)
_CONCRETE = re.compile(
    r"(?:#[0-9a-fA-F]{6}\b|\b\d+(?:mm|°|%|k)?\b|leather|linen|metal|wood|stone|glass|"
    r"camera-left|camera-right|foreground|midground|background)",
    re.IGNORECASE,
)
_LABELS = re.compile(
    r"(?:^|\n)\s*(?:subject|clothing|pose|location|lighting|camera|mood|technical)\s*:",
    re.IGNORECASE,
)


def _field_score(field: str, value: Any) -> float:
    if field == "scene_elements":
        if not isinstance(value, list) or not value:
            return 0.0
        valid = sum(
            isinstance(item, dict)
            and bool(str(item.get("desc", "")).strip())
            and isinstance(item.get("bbox"), list)
            and len(item["bbox"]) == 4
            for item in value
        )
        return round(valid / len(value), 3)
    text = str(value or "").strip()
    if not text:
        return 0.0
    lower = text.casefold()
    score = 0.35
    if len(text) >= 30:
        score += 0.15
    if len(text) >= 70:
        score += 0.15
    if len(text) >= 140:
        score += 0.10
    if _CONCRETE.search(text):
        score += 0.15
    if any(marker in lower for marker in _LIMITATIONS):
        # Honest uncertainty is preferable to invented detail.
        score = max(score, 0.58)
    score -= min(0.35, sum(lower.count(marker) for marker in _HEDGES) * 0.10)
    if field == "reconstruction_prompt" and len(text) < 80:
        score -= 0.20
    return round(max(0.0, min(1.0, score)), 3)


def _expected_person_count(subject: str) -> int | None:
    text = subject.strip().casefold()
    words = {"one": 1, "single": 1, "two": 2, "three": 3, "four": 4}
    for word, value in words.items():
        if re.search(rf"\b{word}\b", text):
            return value
    match = re.search(r"\b([1-4])\s+(?:people|persons|characters|subjects|adults|children)\b", text)
    return int(match.group(1)) if match else None


def build_evidence_report(analysis: dict[str, Any]) -> dict[str, Any]:
    """Build field confidence and cross-field hallucination warnings."""

    field_scores = {field: _field_score(field, analysis.get(field)) for field in _ANALYSIS_FIELDS}
    field_scores["scene_elements"] = _field_score("scene_elements", analysis.get("scene_elements"))
    warnings = []
    for field in _ANALYSIS_FIELDS:
        text = str(analysis.get(field, "") or "")
        lower = text.casefold()
        found = [marker for marker in _HEDGES if marker in lower]
        if found:
            warnings.append(
                {
                    "severity": "minor",
                    "field": field,
                    "message": "hedged/alternate language: " + ", ".join(found),
                }
            )
        if any(marker in lower for marker in _LIMITATIONS) and len(text) > 180:
            warnings.append(
                {
                    "severity": "minor",
                    "field": field,
                    "message": "visibility limitation is mixed with unusually detailed claims",
                }
            )

    reconstruction = str(analysis.get("reconstruction_prompt", "") or "")
    if _LABELS.search(reconstruction) or "```" in reconstruction:
        warnings.append(
            {
                "severity": "critical",
                "field": "reconstruction_prompt",
                "message": "reconstruction prompt contains field labels or Markdown",
            }
        )

    clothing = str(analysis.get("clothing", "") or "").casefold()
    if "fully clothed" in clothing and any(term in clothing for term in ("nude", "naked", "unclothed")):
        warnings.append(
            {
                "severity": "critical",
                "field": "clothing",
                "message": "contradictory clothing state",
            }
        )

    pose = str(analysis.get("pose", "") or "").casefold()
    location = str(analysis.get("location", "") or "").casefold()
    elements = analysis.get("scene_elements", [])
    element_text = " ".join(
        str(item.get("desc", "")) for item in elements if isinstance(item, dict)
    ).casefold()
    support_text = f"{location} {element_text}"
    support_text = re.sub(
        r"(?:no|without|not)\s+(?:visible\s+)?(?:chair|seat|bench|sofa|stool|ground|floor)",
        "",
        support_text,
    )
    if "sitting" in pose and not any(
        term in support_text for term in ("chair", "seat", "bench", "sofa", "stool", "ground", "floor")
    ):
        warnings.append(
            {
                "severity": "major",
                "field": "pose/location",
                "message": "sitting pose has no visible support surface",
            }
        )

    expected_count = _expected_person_count(str(analysis.get("subject", "") or ""))
    person_count = sum(
        str(item.get("type", "")).casefold() in {"person", "character", "human", "subject"}
        for item in elements
        if isinstance(item, dict)
    )
    if expected_count is not None and person_count and expected_count != person_count:
        warnings.append(
            {
                "severity": "major",
                "field": "subject/scene_elements",
                "message": f"subject count {expected_count} conflicts with {person_count} person elements",
            }
        )

    score_values = list(field_scores.values())
    overall = round(sum(score_values) / len(score_values), 3) if score_values else 0.0
    critical_count = sum(item["severity"] == "critical" for item in warnings)
    major_count = sum(item["severity"] == "major" for item in warnings)
    if critical_count:
        overall = min(overall, 0.35)
    elif major_count:
        overall = min(overall, 0.58)
    grade = "high" if overall >= 0.78 else ("medium" if overall >= 0.55 else "low")
    low_fields = [field for field, score in field_scores.items() if score < 0.5]
    return {
        "schema": "omg.image_analysis_quality",
        "version": 1,
        "overall_confidence": overall,
        "grade": grade,
        "field_scores": field_scores,
        "low_confidence_fields": low_fields,
        "warnings": warnings,
        "warning_counts": {
            "critical": critical_count,
            "major": major_count,
            "minor": sum(item["severity"] == "minor" for item in warnings),
        },
        "method": "deterministic heuristic; not a calibrated probability",
    }


def evidence_quality_errors(analysis: dict[str, Any]) -> list[str]:
    report = build_evidence_report(analysis)
    errors = []
    if report["overall_confidence"] < 0.45:
        errors.append(
            f"image analysis evidence confidence is too low ({report['overall_confidence']:.3f})"
        )
    for warning in report["warnings"]:
        if warning["severity"] == "critical":
            errors.append(f"{warning['field']}: {warning['message']}")
    if report["warning_counts"]["major"] >= 2:
        errors.append("image analysis has multiple major cross-field inconsistencies")
    return errors


def collect_analysis_objects(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "subject" in payload:
        return [("analysis", payload)]
    collected = []
    for key in ("frames", "images"):
        for index, item in enumerate(payload.get(key, []), start=1):
            if isinstance(item, dict):
                analysis = item.get("analysis")
                if isinstance(analysis, dict):
                    label_value = item.get("filename") or item.get("frame") or index
                    label = f"{key[:-1]}_{label_value}"
                    collected.append((label, analysis))
    return collected
