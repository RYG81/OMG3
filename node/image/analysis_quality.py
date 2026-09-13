"""Inspect deterministic evidence/confidence diagnostics from image-analysis JSON."""

from __future__ import annotations

import json

from ...tasks.image_evidence import build_evidence_report, collect_analysis_objects


class OllamaAnalysisQualityInspector:
    """Score image-analysis evidence and surface hallucination/cross-field warnings."""

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "inspect_quality"
    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "quality_json",
        "overall_confidence",
        "grade",
        "low_confidence_fields",
        "warnings",
        "reconstruction_prompt",
    )
    DESCRIPTION = (
        "Inspect deterministic field confidence, visibility limitations, hedging, person-count, "
        "pose/support, clothing contradiction, and reconstruction-format diagnostics."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_json": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def inspect_quality(self, analysis_json: str):
        try:
            payload = json.loads(analysis_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"analysis_json is invalid: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("analysis_json must contain a JSON object")
        objects = collect_analysis_objects(payload)
        if not objects:
            raise ValueError("No image-analysis objects were found")
        reports = []
        warnings = []
        low_fields = set()
        reconstruction = ""
        for label, analysis in objects:
            report = build_evidence_report(analysis)
            reports.append({"label": label, "report": report})
            low_fields.update(f"{label}.{field}" for field in report["low_confidence_fields"])
            for warning in report["warnings"]:
                warnings.append({"source": label, **warning})
            if not reconstruction:
                reconstruction = str(analysis.get("reconstruction_prompt", ""))
        overall = round(
            sum(item["report"]["overall_confidence"] for item in reports) / len(reports), 3
        )
        if any(item["severity"] == "critical" for item in warnings):
            overall = min(overall, 0.35)
        grade = "high" if overall >= 0.78 else ("medium" if overall >= 0.55 else "low")
        output = {
            "schema": "omg.image_analysis_quality_collection",
            "version": 1,
            "overall_confidence": overall,
            "grade": grade,
            "analyses": reports,
            "low_confidence_fields": sorted(low_fields),
            "warnings": warnings,
            "method": "deterministic heuristic; not a calibrated probability",
        }
        warning_text = "\n".join(
            f"[{item['severity']}] {item['source']} · {item['field']}: {item['message']}"
            for item in warnings
        )
        return (
            json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True),
            overall,
            grade,
            "\n".join(sorted(low_fields)),
            warning_text,
            reconstruction,
        )
