"""JSON Schema parsing and validation for structured Ollama responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator, SchemaError

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class StructuredResult:
    value: Any | None
    normalized_json: str
    valid: bool
    errors: tuple[str, ...]


def parse_schema(schema_json: str) -> dict[str, Any]:
    """Parse and validate a Draft 2020-12 JSON Schema object."""

    if len(schema_json) > 64 * 1024:
        raise ValueError("JSON Schema exceeds the 64 KB safety limit")
    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON Schema JSON: {exc.msg} at line {exc.lineno}") from exc
    if not isinstance(schema, dict):
        raise ValueError("JSON Schema must be a JSON object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc
    return schema


def extract_json_value(text: str) -> Any:
    """Extract the first complete JSON object/array/value from model text."""

    raw = str(text or "").strip()
    fenced = _FENCE.fullmatch(raw)
    if fenced:
        raw = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(raw) if char in "{["]
    if not starts:
        # Structured schemas can technically return scalar roots.
        starts = [0]
    for start in starts:
        try:
            value, _end = decoder.raw_decode(raw[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("Model response did not contain valid JSON")


def validate_structured_text(text: str, schema: dict[str, Any]) -> StructuredResult:
    """Parse model text and validate it against *schema*."""

    try:
        value = extract_json_value(text)
    except ValueError as exc:
        return StructuredResult(None, "", False, (str(exc),))

    validator = Draft202012Validator(schema)
    validation_errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    errors = []
    for error in validation_errors[:20]:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{path}: {error.message}")
    if len(validation_errors) > 20:
        errors.append(f"... and {len(validation_errors) - 20} more validation errors")

    normalized = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    return StructuredResult(value, normalized, not errors, tuple(errors))


def build_repair_prompt(original_prompt: str, invalid_response: str, errors: tuple[str, ...]) -> str:
    """Create a bounded correction prompt for one schema-repair attempt."""

    response_excerpt = invalid_response[-12_000:]
    error_text = "\n".join(f"- {error}" for error in errors)
    return (
        "Correct the previous response so it satisfies the required JSON Schema and quality "
        "requirements. Return only the corrected JSON value, with no markdown or explanation.\n\n"
        f"Original request:\n{original_prompt}\n\n"
        f"Validation errors:\n{error_text}\n\n"
        f"Previous response:\n{response_excerpt}"
    )
