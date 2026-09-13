"""Bounded local image batches, folder catalogs, and structured text/data loading."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps

from ...utils.path_utils import ensure_within, resolve_input_path
from ...utils.progress import ProgressReporter

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}
TEXT_DATA_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".srt", ".vtt", ".log", ".prompt", ".ini", ".cfg",
}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | TEXT_DATA_EXTENSIONS
FILE_FAMILIES = {
    "images": IMAGE_EXTENSIONS,
    "video": VIDEO_EXTENSIONS,
    "audio": AUDIO_EXTENSIONS,
    "text_data": TEXT_DATA_EXTENSIONS,
    "all_supported": SUPPORTED_EXTENSIONS,
}

_MAX_SCAN_ENTRIES = 10_000
_MAX_LIST_PATHS = 256
_MAX_SOURCE_IMAGE_BYTES = 100 * 1024 * 1024
_MAX_SOURCE_PIXELS = 40_000_000
_MAX_BATCH_PIXELS = 4_194_304
_MAX_REPORTED_ERRORS = 50
_TOKEN_SPLIT = re.compile(r"(\d+)")


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.casefold() for part in _TOKEN_SPLIT.split(path.name)]


def _discover_files(
    folder: Path,
    extensions: set[str],
    recursive: bool,
    filename_filter: str,
) -> tuple[list[Path], bool, list[str]]:
    pattern = "**/*" if recursive else "*"
    filter_text = str(filename_filter or "").strip().casefold()
    candidates = []
    errors = []
    scan_truncated = False
    for scanned, path in enumerate(folder.glob(pattern), start=1):
        if scanned > _MAX_SCAN_ENTRIES:
            scan_truncated = True
            break
        try:
            if not path.is_file() or path.suffix.casefold() not in extensions:
                continue
            if filter_text and filter_text not in path.name.casefold():
                continue
            candidates.append(ensure_within(folder, path))
        except (OSError, ValueError) as exc:
            if len(errors) < _MAX_REPORTED_ERRORS:
                errors.append(f"{path.name}: {exc}")
    candidates.sort(key=lambda path: (_natural_key(path), str(path).casefold()))
    return candidates, scan_truncated, errors


def _workflow_path(path: Path, allow_external_paths: bool) -> str:
    if allow_external_paths:
        return str(path)
    import folder_paths

    input_root = Path(folder_paths.get_input_directory()).resolve(strict=False)
    return str(path.relative_to(input_root)).replace("\\", "/")


def _file_state(paths: list[Path]) -> list[tuple[str, int, int]]:
    state = []
    for path in paths:
        try:
            stat = path.stat()
            state.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            state.append((str(path), -1, -1))
    return state


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_folder(folder_path: str, allow_external_paths: bool) -> Path:
    folder = resolve_input_path(folder_path, allow_external=allow_external_paths)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    return folder


def _parse_path_list(paths: str) -> list[str]:
    raw = str(paths or "").strip()
    if not raw:
        raise ValueError("paths cannot be empty")
    values: list[str]
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"paths JSON is invalid: {exc}") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("paths JSON must be an array of strings")
        values = parsed
    else:
        values = [line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(values) > _MAX_LIST_PATHS:
        raise ValueError(f"At most {_MAX_LIST_PATHS} paths can be supplied")
    return values


def _resolve_image_paths(paths: str, allow_external_paths: bool) -> list[Path]:
    resolved = []
    for value in _parse_path_list(paths):
        path = resolve_input_path(value, allow_external=allow_external_paths)
        if path.suffix.casefold() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix or '(none)'}")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        resolved.append(path)
    return resolved


def _validate_output_size(width: int, height: int) -> int:
    pixels = int(width) * int(height)
    if width < 64 or height < 64 or width > 4096 or height > 4096:
        raise ValueError("target width and height must each be between 64 and 4096")
    if pixels > _MAX_BATCH_PIXELS:
        raise ValueError("one output image exceeds the total batch pixel safety limit")
    return pixels


def _batch_capacity(width: int, height: int, requested: int) -> int:
    pixels = _validate_output_size(width, height)
    return max(1, min(int(requested), _MAX_BATCH_PIXELS // pixels))


def _resample_rgba(
    image: Image.Image,
    width: int,
    height: int,
    resize_mode: str,
    pad_color: tuple[int, int, int],
) -> Image.Image:
    if resize_mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    if resize_mode == "cover_crop":
        return ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    if resize_mode == "contain_pad":
        contained = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
        output = Image.new("RGBA", (width, height), (*pad_color, 255))
        offset = ((width - contained.width) // 2, (height - contained.height) // 2)
        # Paste without a mask so source alpha is preserved while added padding stays opaque.
        output.paste(contained, offset)
        return output
    raise ValueError(f"Unknown resize_mode: {resize_mode}")


def _load_one_image(
    path: Path,
    width: int,
    height: int,
    resize_mode: str,
    pad_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stat = path.stat()
    if stat.st_size > _MAX_SOURCE_IMAGE_BYTES:
        raise ValueError("source image exceeds the 100 MB file limit")
    with Image.open(path) as source:
        source_width, source_height = source.size
        if source_width * source_height > _MAX_SOURCE_PIXELS:
            raise ValueError("source image exceeds the 40-megapixel limit")
        source_format = str(source.format or path.suffix.lstrip(".")).upper()
        source = ImageOps.exif_transpose(source)
        has_alpha = "A" in source.getbands() or "transparency" in source.info
        rgba = source.convert("RGBA")
        resized = _resample_rgba(rgba, width, height, resize_mode, pad_color)
        alpha = np.asarray(resized.getchannel("A"), dtype=np.float32) / 255.0
        background = Image.new("RGBA", resized.size, (*pad_color, 255))
        composited = Image.alpha_composite(background, resized).convert("RGB")
        rgb = np.asarray(composited, dtype=np.float32) / 255.0
    metadata = {
        "path": str(path),
        "filename": path.name,
        "format": source_format,
        "source_width": source_width,
        "source_height": source_height,
        "output_width": width,
        "output_height": height,
        "has_alpha": has_alpha,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    # Match ComfyUI LoadImage semantics: transparent pixels are white in MASK.
    return rgb, 1.0 - alpha, metadata


def _load_image_batch(
    paths: list[Path],
    width: int,
    height: int,
    resize_mode: str,
    pad_color: str,
    max_images: int,
) -> tuple[Any, Any, list[dict[str, Any]], list[str], int]:
    try:
        parsed_color = ImageColor.getrgb(str(pad_color or "#000000"))
        color = tuple(parsed_color[:3])
        if len(color) != 3:
            raise ValueError("pad color must resolve to RGB")
    except ValueError as exc:
        raise ValueError(f"Invalid pad_color: {pad_color}") from exc
    requested = min(len(paths), int(max_images))
    capacity = _batch_capacity(width, height, requested)
    selected = paths[:capacity]
    image_array = np.empty((capacity, height, width, 3), dtype=np.float32)
    mask_array = np.empty((capacity, height, width), dtype=np.float32)
    loaded_metadata = []
    errors = []
    loaded = 0
    progress = ProgressReporter(len(selected) or 1)
    for path in selected:
        progress.check_interrupted()
        try:
            rgb, mask, metadata = _load_one_image(
                path, width, height, resize_mode, color
            )
            image_array[loaded] = rgb
            mask_array[loaded] = mask
            loaded_metadata.append(metadata)
            loaded += 1
        except (OSError, ValueError) as exc:
            if len(errors) < _MAX_REPORTED_ERRORS:
                errors.append(f"{path.name}: {exc}")
        progress.update()
    if not loaded:
        raise ValueError("No readable images were loaded" + (f": {'; '.join(errors)}" if errors else ""))
    images = torch.from_numpy(image_array[:loaded])
    masks = torch.from_numpy(mask_array[:loaded])
    truncated = max(0, len(paths) - len(selected))
    return images, masks, loaded_metadata, errors, truncated


def _image_manifest(
    source: str,
    paths: list[Path],
    metadata: list[dict[str, Any]],
    errors: list[str],
    truncated: int,
    allow_external_paths: bool,
    resize_mode: str,
) -> dict[str, Any]:
    loaded_paths = [
        _workflow_path(Path(item["path"]), allow_external_paths) for item in metadata
    ]
    normalized_metadata = []
    for item, workflow_path in zip(metadata, loaded_paths):
        normalized_item = dict(item)
        normalized_item["path"] = workflow_path
        normalized_metadata.append(normalized_item)
    return {
        "schema": "omg.image_batch_manifest",
        "version": 1,
        "source": source,
        "requested_count": len(paths),
        "loaded_count": len(metadata),
        "paths": loaded_paths,
        "images": normalized_metadata,
        "resize_mode": resize_mode,
        "truncated_for_memory_safety": truncated,
        "errors": errors,
        "limits": {
            "max_source_bytes": _MAX_SOURCE_IMAGE_BYTES,
            "max_source_pixels": _MAX_SOURCE_PIXELS,
            "max_batch_pixels": _MAX_BATCH_PIXELS,
        },
    }


class ComfyUIOMGImageFolderSelect:
    """Load one indexed image from a naturally sorted local input folder."""

    CATEGORY = "ComfyUI-OMG/Loaders"
    FUNCTION = "load_image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "image",
        "mask",
        "current_index",
        "image_count",
        "file_path",
        "manifest_json",
    )
    DESCRIPTION = (
        "Load one clamped, 1-based image from a naturally sorted ComfyUI input folder without "
        "decoding the rest of the folder."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "images"}),
                "image_index": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "resize_mode": (["contain_pad", "cover_crop", "stretch"], {"default": "contain_pad"}),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": False}),
                "filename_filter": ("STRING", {"default": ""}),
                "pad_color": ("STRING", {"default": "#000000"}),
                "allow_external_paths": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        folder_path: str,
        image_index: int,
        target_width: int,
        target_height: int,
        resize_mode: str,
        recursive: bool = False,
        filename_filter: str = "",
        pad_color: str = "#000000",
        allow_external_paths: bool = False,
        **_,
    ):
        try:
            folder = _resolve_folder(folder_path, allow_external_paths)
            paths, truncated, errors = _discover_files(
                folder, IMAGE_EXTENSIONS, recursive, filename_filter
            )
            selected_index = max(1, min(int(image_index), len(paths))) if paths else 0
            state: Any = {
                "selected": _file_state([paths[selected_index - 1]]) if selected_index else [],
                "available_count": len(paths),
            }
        except Exception as exc:
            state, truncated, errors = f"error:{exc}", False, []
        return _fingerprint(
            {
                "state": state,
                "scan_truncated": truncated,
                "scan_errors": errors,
                "folder_path": folder_path,
                "image_index": image_index,
                "recursive": recursive,
                "filename_filter": filename_filter,
                "allow_external_paths": allow_external_paths,
                "width": target_width,
                "height": target_height,
                "resize_mode": resize_mode,
                "pad_color": pad_color,
            }
        )

    def load_image(
        self,
        folder_path: str,
        image_index: int,
        target_width: int,
        target_height: int,
        resize_mode: str,
        recursive: bool = False,
        filename_filter: str = "",
        pad_color: str = "#000000",
        allow_external_paths: bool = False,
    ):
        folder = _resolve_folder(folder_path, allow_external_paths)
        paths, scan_truncated, scan_errors = _discover_files(
            folder, IMAGE_EXTENSIONS, recursive, filename_filter
        )
        if not paths:
            raise ValueError(f"No supported images found in folder: {folder}")
        selected_index = max(1, min(int(image_index), len(paths)))
        selected = paths[selected_index - 1]
        images, masks, metadata, load_errors, _ = _load_image_batch(
            [selected], target_width, target_height, resize_mode, pad_color, 1
        )
        errors = [*scan_errors, *load_errors][:_MAX_REPORTED_ERRORS]
        manifest = _image_manifest(
            "folder_select", [selected], metadata, errors, 0,
            allow_external_paths, resize_mode
        )
        manifest.update(
            folder=str(folder),
            current_index=selected_index,
            available_count=len(paths),
            scan_truncated=scan_truncated,
        )
        return (
            images,
            masks,
            selected_index,
            len(paths),
            manifest["paths"][0],
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        )


class ComfyUIOMGImageFolderLoader:
    """Load a naturally sorted, normalized image batch from a local input folder."""

    CATEGORY = "ComfyUI-OMG/Loaders"
    FUNCTION = "load_folder"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "masks", "image_count", "file_paths", "manifest_json")
    DESCRIPTION = (
        "Load a bounded, naturally sorted image batch from a ComfyUI input folder with alpha "
        "masks, resizing, filtering, stride, metadata, and memory/pixel safety limits."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "images"}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "resize_mode": (["contain_pad", "cover_crop", "stretch"], {"default": "contain_pad"}),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": False}),
                "filename_filter": ("STRING", {"default": ""}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "max_images": ("INT", {"default": 16, "min": 1, "max": 64}),
                "pad_color": ("STRING", {"default": "#000000"}),
                "allow_external_paths": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        folder_path: str,
        target_width: int,
        target_height: int,
        resize_mode: str,
        recursive: bool = False,
        filename_filter: str = "",
        start_index: int = 0,
        stride: int = 1,
        max_images: int = 16,
        pad_color: str = "#000000",
        allow_external_paths: bool = False,
        **_,
    ):
        try:
            folder = _resolve_folder(folder_path, allow_external_paths)
            paths, truncated, errors = _discover_files(
                folder, IMAGE_EXTENSIONS, recursive, filename_filter
            )
            selected = paths[start_index::stride][:max_images]
            state: Any = _file_state(selected)
        except Exception as exc:
            state = f"error:{exc}"
            truncated, errors = False, []
        return _fingerprint(
            {
                "state": state,
                "scan_truncated": truncated,
                "scan_errors": errors,
                "folder_path": folder_path,
                "recursive": recursive,
                "filename_filter": filename_filter,
                "start_index": start_index,
                "stride": stride,
                "max_images": max_images,
                "allow_external_paths": allow_external_paths,
                "width": target_width,
                "height": target_height,
                "resize_mode": resize_mode,
                "pad_color": pad_color,
            }
        )

    def load_folder(
        self,
        folder_path: str,
        target_width: int,
        target_height: int,
        resize_mode: str,
        recursive: bool = False,
        filename_filter: str = "",
        start_index: int = 0,
        stride: int = 1,
        max_images: int = 16,
        pad_color: str = "#000000",
        allow_external_paths: bool = False,
    ):
        folder = _resolve_folder(folder_path, allow_external_paths)
        discovered, scan_truncated, scan_errors = _discover_files(
            folder, IMAGE_EXTENSIONS, recursive, filename_filter
        )
        paths = discovered[int(start_index)::int(stride)]
        if not paths:
            raise ValueError(f"No supported images found in folder: {folder}")
        images, masks, metadata, load_errors, memory_truncated = _load_image_batch(
            paths, target_width, target_height, resize_mode, pad_color, max_images
        )
        errors = [*scan_errors, *load_errors][:_MAX_REPORTED_ERRORS]
        manifest = _image_manifest(
            "folder", paths, metadata, errors, memory_truncated, allow_external_paths, resize_mode
        )
        manifest.update(
            folder=str(folder),
            discovered_count=len(discovered),
            scan_truncated=scan_truncated,
            start_index=int(start_index),
            stride=int(stride),
        )
        file_paths = manifest["paths"]
        return (
            images,
            masks,
            len(metadata),
            "\n".join(file_paths),
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        )


class ComfyUIOMGImageListLoader:
    """Load an ordered image batch from newline or JSON-array file paths."""

    CATEGORY = "ComfyUI-OMG/Loaders"
    FUNCTION = "load_paths"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "masks", "image_count", "file_paths", "manifest_json")
    DESCRIPTION = (
        "Load multiple ordered images from newline-separated paths or a JSON string array, "
        "normalizing them into one bounded ComfyUI image/mask batch."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paths": ("STRING", {"default": "images/frame_001.png", "multiline": True}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "resize_mode": (["contain_pad", "cover_crop", "stretch"], {"default": "contain_pad"}),
            },
            "optional": {
                "max_images": ("INT", {"default": 16, "min": 1, "max": 64}),
                "pad_color": ("STRING", {"default": "#000000"}),
                "allow_external_paths": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        paths: str,
        target_width: int,
        target_height: int,
        resize_mode: str,
        max_images: int = 16,
        pad_color: str = "#000000",
        allow_external_paths: bool = False,
        **_,
    ):
        try:
            resolved = _resolve_image_paths(paths, allow_external_paths)[:max_images]
            state: Any = _file_state(resolved)
        except Exception as exc:
            state = f"error:{exc}"
        return _fingerprint(
            {
                "state": state,
                "paths": paths,
                "max_images": max_images,
                "allow_external_paths": allow_external_paths,
                "width": target_width,
                "height": target_height,
                "resize_mode": resize_mode,
                "pad_color": pad_color,
            }
        )

    def load_paths(
        self,
        paths: str,
        target_width: int,
        target_height: int,
        resize_mode: str,
        max_images: int = 16,
        pad_color: str = "#000000",
        allow_external_paths: bool = False,
    ):
        resolved = _resolve_image_paths(paths, allow_external_paths)
        images, masks, metadata, errors, memory_truncated = _load_image_batch(
            resolved, target_width, target_height, resize_mode, pad_color, max_images
        )
        manifest = _image_manifest(
            "path_list", resolved, metadata, errors, memory_truncated,
            allow_external_paths, resize_mode
        )
        file_paths = manifest["paths"]
        return (
            images,
            masks,
            len(metadata),
            "\n".join(file_paths),
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        )


class ComfyUIOMGFolderCatalog:
    """Catalog supported files without decoding or loading their contents."""

    CATEGORY = "ComfyUI-OMG/Loaders"
    FUNCTION = "catalog"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("file_paths", "manifest_json", "file_count", "total_bytes", "truncated")
    DESCRIPTION = (
        "Create a lightweight local manifest for image, video, audio, or text/data files. "
        "No media is decoded and the path list can feed the Image List Loader."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "file_family": (list(FILE_FAMILIES), {"default": "all_supported"}),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": True}),
                "filename_filter": ("STRING", {"default": ""}),
                "max_files": ("INT", {"default": 250, "min": 1, "max": 2000}),
                "max_total_megabytes": (
                    "FLOAT", {"default": 1024.0, "min": 1.0, "max": 100000.0, "step": 1.0}
                ),
                "allow_external_paths": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            folder = _resolve_folder(
                str(kwargs.get("folder_path", "")), bool(kwargs.get("allow_external_paths", False))
            )
            family = str(kwargs.get("file_family", "all_supported"))
            paths, truncated, errors = _discover_files(
                folder,
                FILE_FAMILIES.get(family, SUPPORTED_EXTENSIONS),
                bool(kwargs.get("recursive", True)),
                str(kwargs.get("filename_filter", "")),
            )
            paths = paths[:int(kwargs.get("max_files", 250))]
            state: Any = _file_state(paths)
        except Exception as exc:
            state, truncated, errors = f"error:{exc}", False, []
        return _fingerprint(
            {"state": state, "truncated": truncated, "errors": errors, "options": kwargs}
        )

    def catalog(
        self,
        folder_path: str,
        file_family: str,
        recursive: bool = True,
        filename_filter: str = "",
        max_files: int = 250,
        max_total_megabytes: float = 1024.0,
        allow_external_paths: bool = False,
    ):
        folder = _resolve_folder(folder_path, allow_external_paths)
        extensions = FILE_FAMILIES.get(file_family)
        if extensions is None:
            raise ValueError(f"Unknown file_family: {file_family}")
        discovered, scan_truncated, errors = _discover_files(
            folder, extensions, recursive, filename_filter
        )
        byte_limit = int(max_total_megabytes * 1024 * 1024)
        entries = []
        total_bytes = 0
        truncated = scan_truncated
        for path in discovered:
            if len(entries) >= int(max_files):
                truncated = True
                break
            try:
                stat = path.stat()
                if total_bytes + stat.st_size > byte_limit:
                    truncated = True
                    if len(errors) < _MAX_REPORTED_ERRORS:
                        errors.append(f"Stopped before {path.name}: total byte limit reached")
                    break
                workflow_path = _workflow_path(path, allow_external_paths)
                entries.append(
                    {
                        "path": workflow_path,
                        "relative_to_folder": str(path.relative_to(folder)).replace("\\", "/"),
                        "filename": path.name,
                        "extension": path.suffix.casefold(),
                        "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
                total_bytes += stat.st_size
            except (OSError, ValueError) as exc:
                if len(errors) < _MAX_REPORTED_ERRORS:
                    errors.append(f"{path.name}: {exc}")
        manifest = {
            "schema": "omg.file_catalog",
            "version": 1,
            "folder": str(folder),
            "file_family": file_family,
            "file_count": len(entries),
            "total_bytes": total_bytes,
            "truncated": truncated,
            "scan_limit": _MAX_SCAN_ENTRIES,
            "files": entries,
            "errors": errors,
            "method": "metadata-only scan; file contents and media are not loaded",
        }
        return (
            "\n".join(item["path"] for item in entries),
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            len(entries),
            total_bytes,
            truncated,
        )


def _decode_text(data: bytes, encoding: str) -> str:
    try:
        return data.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not decode file using {encoding}: {exc}") from exc


def _normalize_data_file(text: str, extension: str, max_rows: int) -> tuple[str, dict[str, Any]]:
    if extension == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON is invalid: {exc}") from exc
        return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), {
            "parsed_type": type(value).__name__,
        }
    if extension == ".jsonl":
        rows = []
        truncated = False
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            if len(rows) >= max_rows:
                truncated = True
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL line {line_number} is invalid: {exc}") from exc
        return json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True), {
            "row_count": len(rows), "rows_truncated": truncated,
        }
    if extension in {".csv", ".tsv"}:
        delimiter = "," if extension == ".csv" else "\t"
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        rows = []
        truncated = False
        for row in reader:
            if len(rows) >= max_rows:
                truncated = True
                break
            rows.append(dict(row))
        return json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True), {
            "row_count": len(rows),
            "columns": list(reader.fieldnames or []),
            "rows_truncated": truncated,
        }
    return "", {"parsed_type": "raw_text"}


class ComfyUIOMGDataFileLoader:
    """Load one bounded text, prompt, subtitle, or structured-data file."""

    CATEGORY = "ComfyUI-OMG/Loaders"
    FUNCTION = "load_file"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "text",
        "normalized_data_json",
        "file_format",
        "filename",
        "size_bytes",
        "metadata_json",
    )
    DESCRIPTION = (
        "Load bounded TXT, Markdown, prompt, JSON/JSONL, CSV/TSV, YAML, XML/HTML, subtitle, "
        "log, INI, or CFG files; JSON and tabular formats receive optional local normalization."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "data/example.json"}),
            },
            "optional": {
                "parse_mode": (["auto", "raw_text", "normalized"], {"default": "auto"}),
                "encoding": (["utf-8", "utf-8-sig", "latin-1"], {"default": "utf-8"}),
                "max_megabytes": (
                    "FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}
                ),
                "max_rows": ("INT", {"default": 5000, "min": 1, "max": 50000}),
                "allow_external_paths": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, file_path: str, **kwargs):
        try:
            path = resolve_input_path(
                file_path, allow_external=bool(kwargs.get("allow_external_paths", False))
            )
            stat = path.stat()
            state: Any = (str(path), stat.st_mtime_ns, stat.st_size)
        except Exception as exc:
            state = f"error:{exc}"
        return _fingerprint({"state": state, "options": kwargs})

    def load_file(
        self,
        file_path: str,
        parse_mode: str = "auto",
        encoding: str = "utf-8",
        max_megabytes: float = 5.0,
        max_rows: int = 5000,
        allow_external_paths: bool = False,
    ):
        path = resolve_input_path(file_path, allow_external=allow_external_paths)
        extension = path.suffix.casefold()
        if extension not in TEXT_DATA_EXTENSIONS:
            raise ValueError(f"Unsupported text/data extension: {extension or '(none)'}")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        stat = path.stat()
        byte_limit = int(float(max_megabytes) * 1024 * 1024)
        if stat.st_size > byte_limit:
            raise ValueError(
                f"File is {stat.st_size} bytes and exceeds the configured {byte_limit}-byte limit"
            )
        text = _decode_text(path.read_bytes(), encoding)
        normalized = ""
        parse_metadata: dict[str, Any] = {"parsed_type": "raw_text"}
        structured_extension = extension in {".json", ".jsonl", ".csv", ".tsv"}
        if parse_mode == "normalized" or (parse_mode == "auto" and structured_extension):
            normalized, parse_metadata = _normalize_data_file(text, extension, int(max_rows))
        metadata = {
            "schema": "omg.loaded_data_file",
            "version": 1,
            "path": str(path),
            "filename": path.name,
            "format": extension.lstrip("."),
            "encoding": encoding,
            "bytes": stat.st_size,
            "characters": len(text),
            "parse_mode": parse_mode,
            **parse_metadata,
        }
        return (
            text,
            normalized,
            extension.lstrip("."),
            path.name,
            stat.st_size,
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True),
        )


NODE_CLASS_MAPPINGS = {
    "ComfyUIOMGImageFolderSelect": ComfyUIOMGImageFolderSelect,
    "ComfyUIOMGImageFolderLoader": ComfyUIOMGImageFolderLoader,
    "ComfyUIOMGImageListLoader": ComfyUIOMGImageListLoader,
    "ComfyUIOMGFolderCatalog": ComfyUIOMGFolderCatalog,
    "ComfyUIOMGDataFileLoader": ComfyUIOMGDataFileLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIOMGImageFolderSelect": "Image From Folder",
    "ComfyUIOMGImageFolderLoader": "Image Folder Batch Loader",
    "ComfyUIOMGImageListLoader": "Multiple Image Paths Loader",
    "ComfyUIOMGFolderCatalog": "Folder File Catalog",
    "ComfyUIOMGDataFileLoader": "Text & Data File Loader",
}
