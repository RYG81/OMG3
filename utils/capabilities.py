"""Normalize capability metadata returned by Ollama's model-info endpoint."""

from __future__ import annotations

from typing import Any


def require_declared_capability(model_profile: dict, capability: str) -> None:
    """Reject only when Ollama explicitly declares that a capability is absent."""

    report = model_profile.get("_capabilities", {})
    declared = report.get("evidence", {}).get("capabilities_from_server", False)
    key = f"supports_{capability}"
    if declared and not report.get(key, False):
        raise ValueError(
            f"Model {model_profile.get('model', '')!r} does not report {capability} capability"
        )


def build_capability_report(model: str, info: dict[str, Any], is_loaded: bool) -> dict[str, Any]:
    """Return conservative, evidence-based capability flags for one model."""

    raw_capabilities = info.get("capabilities", [])
    if not isinstance(raw_capabilities, list):
        raw_capabilities = []
    declared = {str(item).strip().lower() for item in raw_capabilities if str(item).strip()}

    details = info.get("details", {}) if isinstance(info.get("details"), dict) else {}
    model_info = info.get("model_info", {}) if isinstance(info.get("model_info"), dict) else {}
    families = details.get("families", [])
    if not isinstance(families, list):
        families = []
    family_tokens = {
        str(details.get("family", "")).lower(),
        *(str(item).lower() for item in families),
    }
    family_text = " ".join(sorted(family_tokens))
    model_info_keys = " ".join(str(key).lower() for key in model_info)

    embedding_markers = ("bert", "embed", "embedding")
    supports_embeddings = bool(
        {"embedding", "embeddings"} & declared
        or any(marker in family_text for marker in embedding_markers)
        or any(marker in model.lower() for marker in ("embed", "embedding"))
    )
    supports_vision = bool(
        {"vision", "images", "image"} & declared
        or ".vision." in model_info_keys
        or "clip." in model_info_keys
    )
    supports_tools = bool({"tools", "tool"} & declared)
    supports_thinking = bool({"thinking", "think"} & declared)
    supports_completion = bool("completion" in declared or not supports_embeddings)

    context_length = 0
    context_candidates = []
    for key, value in model_info.items():
        normalized_key = str(key).lower()
        if normalized_key.endswith(".context_length") or normalized_key in {
            "context_length",
            "num_ctx",
        }:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                continue
            if candidate > 0:
                context_candidates.append(candidate)
    if context_candidates:
        context_length = max(context_candidates)

    return {
        "model": model,
        "is_loaded": bool(is_loaded),
        "declared_capabilities": sorted(declared),
        "supports_completion": supports_completion,
        "supports_vision": supports_vision,
        "supports_tools": supports_tools,
        "supports_thinking": supports_thinking,
        "supports_embeddings": supports_embeddings,
        "context_length": context_length,
        "family": details.get("family", "unknown"),
        "families": families,
        "format": details.get("format", "unknown"),
        "parameter_size": details.get("parameter_size", "unknown"),
        "quantization_level": details.get("quantization_level", "unknown"),
        "evidence": {
            "capabilities_from_server": bool(declared),
            "context_keys": sorted(
                str(key)
                for key in model_info
                if str(key).lower().endswith(".context_length")
            ),
        },
    }
