"""
image_utils.py — Image ↔ base64 conversion for Ollama vision models.
"""
from __future__ import annotations

import base64
import hashlib
import io

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def tensor_to_base64(image_tensor, max_size: int = 1024) -> str:
    """
    Convert a ComfyUI IMAGE tensor (B,H,W,C float32 0-1) to a base64 JPEG string.
    Resizes if either dimension exceeds max_size (preserving aspect ratio).
    """
    if Image is None:
        raise ImportError("Pillow is required for image processing")

    # Handle batch dimension — take first image
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]

    # Convert to numpy uint8 without wrapping values outside ComfyUI's 0-1 range.
    raw = image_tensor.cpu().numpy() if hasattr(image_tensor, "cpu") else np.asarray(image_tensor)
    img_np = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    # Resize if too large
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    return pil_to_base64(pil_img)


def mask_to_base64(mask_tensor, max_size: int = 1024) -> str:
    """Convert a ComfyUI MASK tensor to a crisp base64 PNG visualization."""

    if Image is None:
        raise ImportError("Pillow is required for mask processing")
    value = mask_tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value)
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] not in {1, 3, 4}:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D ComfyUI mask, got shape {arr.shape}")
    mask_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    image = Image.fromarray(mask_u8, mode="L")
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.NEAREST,
        )
    return pil_to_base64_png(image)


def image_tensor_hash(image_tensor) -> str:
    """Return a stable hash for ComfyUI IMAGE tensor pixel data."""
    if hasattr(image_tensor, "detach"):
        image_tensor = image_tensor.detach()
    if hasattr(image_tensor, "cpu"):
        image_tensor = image_tensor.cpu()
    if hasattr(image_tensor, "numpy"):
        image_tensor = image_tensor.numpy()

    arr = np.ascontiguousarray(image_tensor)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def pil_to_base64(pil_img, quality: int = 85) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_base64_png(pil_img) -> str:
    """Convert a PIL Image to a lossless base64-encoded PNG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
