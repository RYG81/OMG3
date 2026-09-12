"""Quality checks for free-form core generation nodes."""

from __future__ import annotations

import json


def normalize_generated_output(raw: str, output_format: str, label: str) -> str:
    """Reject empty output and normalize requested JSON output locally."""

    text = str(raw or "").strip()
    if not text:
        raise RuntimeError(f"{label} returned an empty response")
    if output_format != "json":
        return text
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{label} requested JSON but returned invalid JSON: {exc.msg} at line {exc.lineno}"
        ) from exc
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
