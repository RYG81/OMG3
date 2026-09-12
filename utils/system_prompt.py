"""Composable global and per-node custom system-prompt support."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from typing import Any

MAX_CUSTOM_SYSTEM_PROMPT_CHARS = 16_000

# These nodes consume OLLAMA_MODEL but do not send a generative system prompt.
NON_GENERATIVE_OLLAMA_NODE_IDS = frozenset(
    {
        "OllamaEmbeddings",
        "OllamaModelInspector",
        "OllamaRAGIndexBuild",
        "OllamaRAGSearch",
        # Nodes that already have explicit system_prompt as single input - prevent extra custom_system_prompt auto-add per user request "no extra inputs for system prompt other than one asked"
        "OllamaDatabaseGenerator",
        "H3PromptGeneratorOllamaSimple",
        "H3PromptGeneratorClipSimple",
        "OllamaLTX25Prompt",
        "OllamaLTXVVideoPromptImproved",
        "OllamaWanVideoPromptImproved",
        "CookbookStyleAutoFiller",
        "CookbookDynamicInputs",
        "OllamaTrendingPromptLoader",
        "OllamaTrendingPromptFiller",
        "OllamaPromptBuilderWithTrending",
        "OllamaPhotosetFolderAnalyzer",
        "OllamaImageSequenceAnalyzer",
    }
)

_CUSTOM_INPUT_SPEC = (
    "STRING",
    {
        "default": "",
        "multiline": True,
        "tooltip": (
            "Optional node-specific system instructions appended after the built-in prompt "
            "and the Model Loader global custom system prompt."
        ),
    },
)


def normalize_custom_system_prompt(value: str, label: str = "custom_system_prompt") -> str:
    """Validate and trim one user-supplied system-prompt addition."""

    text = str(value or "").strip()
    if len(text) > MAX_CUSTOM_SYSTEM_PROMPT_CHARS:
        raise ValueError(
            f"{label} exceeds the {MAX_CUSTOM_SYSTEM_PROMPT_CHARS:,}-character safety limit"
        )
    if "\x00" in text:
        raise ValueError(f"{label} cannot contain NUL characters")
    return text


def custom_system_prompt_hash(value: str) -> str:
    text = normalize_custom_system_prompt(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def compose_system_prompt(
    defined_system_prompt: str,
    model_profile: dict[str, Any] | None = None,
    node_custom_system_prompt: str = "",
) -> str:
    """Append global and node-specific instructions after the defined system prompt."""

    profile = model_profile if isinstance(model_profile, dict) else {}
    defined = str(defined_system_prompt or "").strip()
    global_custom = normalize_custom_system_prompt(
        str(profile.get("_custom_system_prompt", "") or ""),
        "Model Loader custom_system_prompt",
    )
    profile_node_custom = str(profile.get("_node_custom_system_prompt", "") or "")
    explicit_node_custom = normalize_custom_system_prompt(
        node_custom_system_prompt, "node custom_system_prompt"
    )
    node_custom = explicit_node_custom or normalize_custom_system_prompt(
        profile_node_custom, "node custom_system_prompt"
    )

    sections = []
    if defined:
        sections.append(defined)
    if global_custom and global_custom != defined:
        sections.append("Additional global system instructions:\n" + global_custom)
    if node_custom and node_custom not in {defined, global_custom}:
        sections.append("Additional node-specific system instructions:\n" + node_custom)
    return "\n\n".join(sections)


def _wrap_is_changed(node_class: type) -> None:
    original = getattr(node_class, "IS_CHANGED", None)
    if not callable(original) or getattr(node_class, "__comfy_omg_system_is_changed__", False):
        return
    signature = inspect.signature(original)
    accepts_custom = "custom_system_prompt" in signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    def is_changed(_cls, *args, custom_system_prompt: str = "", **kwargs):
        custom = normalize_custom_system_prompt(custom_system_prompt)
        forwarded = dict(kwargs)
        if accepts_custom or accepts_kwargs:
            forwarded["custom_system_prompt"] = custom
        base_value = original(*args, **forwarded)
        if isinstance(base_value, float) and math.isnan(base_value):
            return base_value
        payload = json.dumps(
            {"base": base_value, "custom_system_prompt_sha256": custom_system_prompt_hash(custom)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    node_class.IS_CHANGED = classmethod(is_changed)
    node_class.__comfy_omg_system_is_changed__ = True


def _install_for_class(node_class: type) -> None:
    if getattr(node_class, "__comfy_omg_custom_system_input__", False):
        return
    function_name = str(getattr(node_class, "FUNCTION", ""))
    original_method = getattr(node_class, function_name)
    method_signature = inspect.signature(original_method)
    if "custom_system_prompt" in method_signature.parameters:
        # This class already declares and handles its own node-specific prompt.
        _wrap_is_changed(node_class)
        node_class.__comfy_omg_custom_system_input__ = True
        return

    original_input_types = node_class.INPUT_TYPES

    def input_types(_cls):
        schema = copy.deepcopy(original_input_types())
        optional = schema.setdefault("optional", {})
        optional.setdefault("custom_system_prompt", copy.deepcopy(_CUSTOM_INPUT_SPEC))
        return schema

    def method(self, *args, custom_system_prompt: str = "", **kwargs):
        custom = normalize_custom_system_prompt(custom_system_prompt)
        if not custom:
            return original_method(self, *args, **kwargs)
        bound = method_signature.bind_partial(self, *args, **kwargs)
        profile = bound.arguments.get("ollama_model")
        if not isinstance(profile, dict):
            return original_method(self, *args, **kwargs)
        profile_with_custom = dict(profile)
        profile_with_custom["_node_custom_system_prompt"] = custom
        bound.arguments["ollama_model"] = profile_with_custom
        return original_method(*bound.args, **bound.kwargs)

    input_types.__name__ = getattr(original_input_types, "__name__", "INPUT_TYPES")
    method.__name__ = getattr(original_method, "__name__", function_name)
    method.__doc__ = getattr(original_method, "__doc__", None)
    node_class.INPUT_TYPES = classmethod(input_types)
    setattr(node_class, function_name, method)
    _wrap_is_changed(node_class)
    node_class.__comfy_omg_custom_system_input__ = True


def install_custom_system_prompt_inputs(node_mappings: dict[str, type]) -> set[str]:
    """Add a per-node textbox to every generative OLLAMA_MODEL consumer."""

    enabled = set()
    for node_id, node_class in node_mappings.items():
        if node_id in NON_GENERATIVE_OLLAMA_NODE_IDS:
            continue
        function_name = str(getattr(node_class, "FUNCTION", ""))
        method = getattr(node_class, function_name, None)
        if not callable(method):
            continue
        if "ollama_model" not in inspect.signature(method).parameters:
            continue
        _install_for_class(node_class)
        enabled.add(node_id)
    return enabled
