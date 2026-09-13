"""Inspect fast deterministic consistency and drift across image analyses."""

from __future__ import annotations

import json

from ...tasks.image_consistency import build_consistency_report
from ...tasks.image_evidence import collect_analysis_objects


class OllamaMultiImageConsistencyInspector:
    """Compare identity, wardrobe, scene, camera, lighting, palette, and style drift."""

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "inspect_consistency"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "consistency_json",
        "consistency_score",
        "drift_score",
        "status",
        "field_summary",
        "conflicts",
    )
    DESCRIPTION = (
        "Compare structured analyses for identity, clothing, environment, camera, lighting, "
        "palette, and style drift using bounded local heuristics with no extra Ollama call."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Aggregate JSON from Image Sequence or Photoset Folder Analyzer",
                    },
                ),
            }
        }

    def inspect_consistency(self, analysis_json: str):
        try:
            payload = json.loads(analysis_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"analysis_json is invalid: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("analysis_json must contain a JSON object")
        objects = collect_analysis_objects(payload)
        if not objects:
            raise ValueError("No image-analysis objects were found")
        report = build_consistency_report(objects)
        field_summary = "\n".join(
            f"{field}: {details['classification']} · drift {details['drift_score']:.3f}"
            for field, details in report["fields"].items()
        )
        conflicts = "\n".join(
            f"[{item['severity']}] {item['field']}.{item['category']}: {item['message']}"
            for item in report["conflicts"]
        )
        return (
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
            report["overall_consistency"],
            report["overall_drift"],
            report["status"],
            field_summary,
            conflicts,
        )
