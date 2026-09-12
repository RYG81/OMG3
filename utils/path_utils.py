"""Safe filesystem helpers for ComfyUI input/output/temp directories."""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def ensure_within(base: str | Path, candidate: str | Path) -> Path:
    """Resolve *candidate* and require it to remain under *base*."""

    base_path = _resolved(base)
    candidate_path = _resolved(candidate)
    if candidate_path != base_path and base_path not in candidate_path.parents:
        raise ValueError(f"Path must stay inside {base_path}")
    return candidate_path


def safe_join(base: str | Path, user_path: str, *, create_parent: bool = False) -> Path:
    """Join an untrusted relative path beneath *base* and block traversal."""

    raw = str(user_path or "").strip().strip('"')
    if not raw:
        raise ValueError("Path cannot be empty")
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError("Absolute paths are not allowed here")
    candidate = ensure_within(base, _resolved(base) / relative)
    if create_parent:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def resolve_input_path(user_path: str, *, allow_external: bool = False) -> Path:
    """Resolve a path under ComfyUI input, unless external access is explicit."""

    import folder_paths

    raw = str(user_path or "").strip().strip('"')
    if not raw:
        raise ValueError("Input path cannot be empty")
    path = Path(raw).expanduser()
    if path.is_absolute():
        if not allow_external:
            raise ValueError("Absolute paths require allow_external_paths=True")
        return path.resolve(strict=False)
    return safe_join(folder_paths.get_input_directory(), raw)


def resolve_output_path(user_path: str, *, create_parent: bool = True) -> Path:
    """Resolve an output filename beneath ComfyUI's output directory."""

    import folder_paths

    return safe_join(folder_paths.get_output_directory(), user_path, create_parent=create_parent)


def sanitize_filename(value: str, *, fallback: str = "output", max_length: int = 120) -> str:
    """Create a portable filename component without directory separators."""

    name = Path(str(value or "")).name.strip()
    name = _SAFE_NAME.sub("_", name).strip(" ._")
    return (name or fallback)[:max_length]
