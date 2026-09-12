"""Shared execution engine for schema-constrained Ollama tasks."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from ..ollama_client import OllamaError, generate
from ..utils.structured_output import (
    StructuredResult,
    build_repair_prompt,
    validate_structured_text,
)
from .cache import TASK_RESPONSE_CACHE
from ..utils.system_prompt import compose_system_prompt
from .quality import QUALITY_RULES_VERSION, validate_task_output
from .sanitization import SANITIZATION_RULES_VERSION, sanitize_structured_output

CachePolicy = str


@dataclass(frozen=True)
class TaskExecution:
    structured: StructuredResult
    raw_output: str
    report: str
    provenance: dict[str, Any]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_text(canonical)


def _apply_quality(
    task_id: str,
    result: StructuredResult,
    request_prompt: str,
    system_prompt: str,
    quality_context: dict[str, Any],
) -> StructuredResult:
    if not result.valid:
        return result
    errors = validate_task_output(
        task_id,
        result.value,
        request_prompt=request_prompt,
        system_prompt=system_prompt,
        context=quality_context,
    )
    if not errors:
        return result
    return StructuredResult(
        result.value,
        result.normalized_json,
        False,
        tuple(f"quality: {error}" for error in errors),
    )


def _apply_sanitization(
    result: StructuredResult,
) -> tuple[StructuredResult, tuple[dict[str, Any], ...], int]:
    if not result.valid:
        return result, (), 0
    sanitized = sanitize_structured_output(result.value)
    if not sanitized.change_count:
        return result, (), 0
    normalized = json.dumps(sanitized.value, indent=2, ensure_ascii=False, sort_keys=True)
    return (
        StructuredResult(sanitized.value, normalized, True, ()),
        sanitized.changes,
        sanitized.change_count,
    )


def _provenance(
    *,
    task_id: str,
    template_version: str,
    model_profile: dict,
    prompt: str,
    system: str,
    schema: dict[str, Any],
    options: dict[str, Any],
    fingerprint: str,
    cache_policy: str,
    cache_hit: bool,
    attempts: int,
    valid: bool,
    duration_ms: int,
    image_hashes: list[str],
) -> dict[str, Any]:
    return {
        "schema": "omg.provenance",
        "version": 1,
        "task_id": task_id,
        "template_version": template_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_profile.get("model", ""),
        "endpoint": model_profile.get("base_url", ""),
        "request_fingerprint": fingerprint,
        "prompt_sha256": _sha256_text(prompt),
        "system_sha256": _sha256_text(system),
        "schema_sha256": _request_fingerprint(schema),
        "image_sha256": image_hashes,
        "options": options,
        "cache_policy": cache_policy,
        "cache_hit": cache_hit,
        "attempts": attempts,
        "valid": valid,
        "duration_ms": duration_ms,
    }


def run_structured_task(
    *,
    task_id: str,
    template_version: str,
    model_profile: dict,
    prompt: str,
    system: str,
    schema: dict[str, Any],
    num_predict: int = 1024,
    temperature: float = 0.0,
    repair_once: bool = True,
    cache_policy: CachePolicy = "use",
    images: list[str] | None = None,
    quality_context: dict[str, Any] | None = None,
    generate_fn: Callable[..., str] = generate,
    think: Any = None,
    filter_thinking: bool = True,
) -> TaskExecution:
    """Execute, validate, optionally repair, cache, and describe one task."""

    if cache_policy not in {"use", "refresh", "bypass"}:
        raise ValueError("cache_policy must be use, refresh, or bypass")
    system = compose_system_prompt(system, model_profile)
    capabilities = model_profile.get("_capabilities", {})
    if capabilities and not capabilities.get("supports_completion", True):
        raise ValueError(f"Model {model_profile.get('model', '')!r} is not a completion model")

    options = {
        "temperature": float(temperature),
        "top_p": model_profile.get("top_p", 0.9),
        "top_k": model_profile.get("top_k", 40),
        "repeat_penalty": model_profile.get("repeat_penalty", 1.1),
        "num_predict": int(num_predict),
        "num_ctx": model_profile.get("num_ctx", 8192),
        "seed": model_profile.get("seed", -1),
        "keep_alive": model_profile.get("keep_alive", "5m"),
        "repair_once": bool(repair_once),
        "quality_rules_version": QUALITY_RULES_VERSION,
        "sanitization_rules_version": SANITIZATION_RULES_VERSION,
        "think": think if think is not None else model_profile.get("think", False),
        "filter_thinking": filter_thinking if filter_thinking is not None else model_profile.get("filter_thinking", True),
    }
    image_hashes = [_sha256_text(image) for image in (images or [])]
    quality_context = dict(quality_context or {})
    fingerprint_payload = {
        "task_id": task_id,
        "template_version": template_version,
        "endpoint": model_profile.get("base_url", ""),
        "model": model_profile.get("model", ""),
        "prompt": prompt,
        "system": system,
        "schema": schema,
        "image_sha256": image_hashes,
        "quality_context": quality_context,
        "options": options,
    }
    fingerprint = _request_fingerprint(fingerprint_payload)
    started = time.monotonic()

    if cache_policy == "use":
        cached = TASK_RESPONSE_CACHE.get(fingerprint)
        if isinstance(cached, TaskExecution):
            duration_ms = round((time.monotonic() - started) * 1000)
            provenance = dict(cached.provenance)
            provenance.update(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                cache_policy=cache_policy,
                cache_hit=True,
                duration_ms=duration_ms,
            )
            return replace(cached, provenance=provenance, report="Valid response loaded from memory cache")

    common = {
        "base_url": model_profile["base_url"],
        "model": model_profile["model"],
        "system": system,
        "top_p": options["top_p"],
        "top_k": options["top_k"],
        "repeat_penalty": options["repeat_penalty"],
        "num_predict": options["num_predict"],
        "num_ctx": options["num_ctx"],
        "seed": options["seed"],
        "response_format": schema,
        "think": options.get("think", False),
        "filter_thinking": options.get("filter_thinking", True),
        "keep_alive": options["keep_alive"],
    }
    if images:
        common["images"] = images
    attempts = 1
    try:
        first_raw = generate_fn(prompt=prompt, temperature=temperature, **common)
    except OllamaError as exc:
        raise RuntimeError(f"{task_id} failed: {exc}") from exc

    parsed_first = validate_structured_text(first_raw, schema)
    if task_id == "structured_generate":
        sanitized_first, first_changes, first_change_count = parsed_first, (), 0
    else:
        sanitized_first, first_changes, first_change_count = _apply_sanitization(parsed_first)
    first = _apply_quality(
        task_id,
        sanitized_first,
        prompt,
        system,
        quality_context,
    )
    sanitization_changes = list(first_changes)
    sanitization_change_count = first_change_count
    final = first
    raw_trace = first_raw
    if first.valid and first_change_count:
        report = (
            "Valid on first response after safe local sanitization "
            f"({first_change_count} field(s) changed)"
        )
    else:
        report = "Valid on first response" if first.valid else "\n".join(first.errors)

    if not first.valid and repair_once:
        attempts = 2
        repair_source = sanitized_first.normalized_json if first_change_count else first_raw
        repair_prompt = build_repair_prompt(prompt, repair_source, first.errors)
        try:
            repaired_raw = generate_fn(
                prompt=repair_prompt,
                temperature=min(float(temperature), 0.1),
                **common,
            )
        except OllamaError as exc:
            report = f"Initial validation failed:\n{report}\n\nRepair request failed: {exc}"
        else:
            parsed_repaired = validate_structured_text(repaired_raw, schema)
            if task_id == "structured_generate":
                sanitized_repaired, repair_changes, repair_change_count = parsed_repaired, (), 0
            else:
                sanitized_repaired, repair_changes, repair_change_count = _apply_sanitization(
                    parsed_repaired
                )
            sanitization_changes.extend(repair_changes)
            sanitization_change_count += repair_change_count
            repaired = _apply_quality(
                task_id,
                sanitized_repaired,
                prompt,
                system,
                quality_context,
            )
            raw_trace = (
                f"--- INITIAL RESPONSE ---\n{first_raw}\n\n"
                f"--- REPAIR RESPONSE ---\n{repaired_raw}"
            )
            if repaired.valid:
                final = repaired
                report = "Valid after one repair. Initial errors:\n" + "\n".join(first.errors)
                if sanitization_change_count:
                    report += (
                        "\n\nSafe local sanitization changed "
                        f"{sanitization_change_count} field(s) across the response trace."
                    )
            else:
                final = repaired
                report = (
                    "Initial validation failed:\n"
                    + "\n".join(first.errors)
                    + "\n\nRepair validation failed:\n"
                    + "\n".join(repaired.errors)
                )

    duration_ms = round((time.monotonic() - started) * 1000)
    provenance = _provenance(
        task_id=task_id,
        template_version=template_version,
        model_profile=model_profile,
        prompt=prompt,
        system=system,
        schema=schema,
        options=options,
        fingerprint=fingerprint,
        cache_policy=cache_policy,
        cache_hit=False,
        attempts=attempts,
        valid=final.valid,
        duration_ms=duration_ms,
        image_hashes=image_hashes,
    )
    if final.errors:
        provenance["validation_errors"] = list(final.errors)
    if sanitization_change_count:
        provenance["sanitization"] = {
            "version": SANITIZATION_RULES_VERSION,
            "change_count": sanitization_change_count,
            "changes": sanitization_changes[:64],
            "raw_response_preserved": True,
        }
    execution = TaskExecution(final, raw_trace, report, provenance)
    if final.valid and cache_policy != "bypass":
        TASK_RESPONSE_CACHE.put(
            fingerprint,
            execution,
            len(raw_trace.encode("utf-8")) + len(final.normalized_json.encode("utf-8")),
        )
    return execution


def get_task_cache_stats() -> dict[str, int]:
    stats = TASK_RESPONSE_CACHE.stats()
    return {
        "entries": stats.entries,
        "bytes": stats.bytes,
        "hits": stats.hits,
        "misses": stats.misses,
        "max_entries": stats.max_entries,
        "max_bytes": stats.max_bytes,
    }


def clear_task_cache() -> int:
    return TASK_RESPONSE_CACHE.clear()
