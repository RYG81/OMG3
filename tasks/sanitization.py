"""Conservative local cleanup for directly usable generated text fields.

Only surface formatting that can be removed without inventing or paraphrasing content is
changed. Raw model responses remain available through the task execution trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SANITIZATION_RULES_VERSION = "1"
_MAX_RECORDED_CHANGES = 64

_FULL_FENCE = re.compile(
    r"^```(?:json|text|markdown|prompt)?\s*\n?(.*?)\n?\s*```$",
    re.IGNORECASE | re.DOTALL,
)
_LABEL_PREFIX = re.compile(
    r"^\s*(?:positive prompt|negative prompt|scene prompt|enhanced prompt|translated prompt|"
    r"edit prompt|video prompt|reconstruction prompt|output|result|answer)\s*:\s*",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_DIRECT_NAMES = {"story", "full_description", "master_sheet"}


@dataclass(frozen=True)
class SanitizationResult:
    value: Any
    change_count: int
    changes: tuple[dict[str, Any], ...]


def _is_direct_field(name: str) -> bool:
    return name == "prompt" or name.endswith("_prompt") or name in _DIRECT_NAMES


def _deduplicate_sentences(text: str) -> tuple[str, bool]:
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    if len(sentences) < 2:
        return text, False
    kept = []
    seen = set()
    changed = False
    for sentence in sentences:
        signature = " ".join(sentence.casefold().split())
        if signature in seen:
            changed = True
            continue
        seen.add(signature)
        kept.append(sentence)
    return (" ".join(kept), True) if changed else (text, False)


def sanitize_direct_text(value: str) -> tuple[str, tuple[str, ...]]:
    """Remove only safe wrappers/duplication from one direct-output string."""

    text = value.strip()
    actions = []
    if text != value:
        actions.append("trimmed_outer_whitespace")

    fenced = _FULL_FENCE.fullmatch(text)
    if fenced:
        inner = fenced.group(1).strip()
        if inner and "```" not in inner:
            text = inner
            actions.append("removed_full_code_fence")

    unlabelled = _LABEL_PREFIX.sub("", text, count=1).strip()
    if unlabelled and unlabelled != text:
        text = unlabelled
        actions.append("removed_output_label")

    deduplicated, changed = _deduplicate_sentences(text)
    if changed:
        text = deduplicated
        actions.append("removed_exact_duplicate_sentence")
    return text, tuple(actions)


def sanitize_structured_output(value: Any) -> SanitizationResult:
    """Return a sanitized copy plus bounded content-free change metadata."""

    recorded: list[dict[str, Any]] = []
    change_count = 0

    def walk(item: Any, path: tuple[str | int, ...], field_name: str = "") -> Any:
        nonlocal change_count
        if isinstance(item, dict):
            return {
                key: walk(child, (*path, str(key)), str(key))
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [walk(child, (*path, index), field_name) for index, child in enumerate(item)]
        if isinstance(item, str) and _is_direct_field(field_name):
            cleaned, actions = sanitize_direct_text(item)
            if actions:
                change_count += 1
                if len(recorded) < _MAX_RECORDED_CHANGES:
                    recorded.append(
                        {
                            "path": ".".join(str(part) for part in path),
                            "actions": list(actions),
                        }
                    )
            return cleaned
        return item

    sanitized = walk(value, ())
    return SanitizationResult(sanitized, change_count, tuple(recorded))
