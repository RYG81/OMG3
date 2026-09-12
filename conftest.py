"""Pytest bootstrap for environments without ComfyUI's torch dependency."""

from __future__ import annotations

import sys
import types

try:
    import torch as _torch  # noqa: F401
except ImportError:
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = type("Tensor", (), {})
    fake_torch.float32 = "float32"
    sys.modules["torch"] = fake_torch
