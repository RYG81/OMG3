"""Small, read-only dashboard API for the ComfyUI-OMG frontend extension."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import time
from urllib.parse import urlsplit

from .ollama_client import (
    OllamaError,
    get_installed_models,
    get_model_info,
    get_running_models,
    normalize_base_url,
)
from .tasks.engine import clear_task_cache, get_task_cache_stats
from .utils.capabilities import build_capability_report
from .tasks.rag import clear_embedding_cache, get_embedding_cache_stats

_ROUTES_REGISTERED = False
_STATUS = {"version": "unknown", "node_count": 0, "stability_tiers": {}}


def dashboard_ollama_url(base_url: str) -> str:
    """Allow loopback dashboard probes unless a host is explicitly allowlisted.

    This restriction applies only to browser dashboard routes. Workflow nodes can
    still use any explicitly configured Ollama URL.
    """

    normalized = normalize_base_url(base_url)
    hostname = (urlsplit(normalized).hostname or "").casefold()
    allowed = {
        host.strip().casefold()
        for host in os.getenv("COMFY_OMG_ALLOWED_OLLAMA_HOSTS", "").split(",")
        if host.strip()
    }
    if "*" in allowed or hostname in allowed or hostname == "localhost":
        return normalized
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and address.is_loopback:
        return normalized
    raise ValueError(
        "Dashboard probes are limited to loopback Ollama by default. Add the exact remote hostname "
        "to COMFY_OMG_ALLOWED_OLLAMA_HOSTS to enable it."
    )


def _ollama_snapshot(base_url: str) -> dict:
    started = time.monotonic()
    installed = get_installed_models(base_url)
    running = get_running_models(base_url)
    running_names = {
        str(item.get("name") or item.get("model") or "") for item in running if isinstance(item, dict)
    }
    models = []
    for item in installed:
        details = item.get("details", {}) if isinstance(item.get("details"), dict) else {}
        name = str(item.get("name") or item.get("model") or "")
        models.append(
            {
                "name": name,
                "size": int(item.get("size", 0) or 0),
                "family": details.get("family", ""),
                "parameter_size": details.get("parameter_size", ""),
                "quantization": details.get("quantization_level", ""),
                "running": name in running_names,
            }
        )
    return {
        "ok": True,
        "base_url": base_url,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "models": models,
        "running_count": len(running),
    }


def _model_capability_snapshot(base_url: str, model: str) -> dict:
    name = str(model or "").strip()
    if not name or len(name) > 256 or any(ord(char) < 32 for char in name):
        raise ValueError("A valid model name is required")
    started = time.monotonic()
    info = get_model_info(base_url, name)
    running = get_running_models(base_url)
    is_loaded = any(
        str(item.get("name") or item.get("model") or "") == name
        for item in running
        if isinstance(item, dict)
    )
    report = build_capability_report(name, info, is_loaded)
    return {
        "ok": True,
        "base_url": base_url,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "capabilities": report,
    }


def _cache_snapshot() -> dict:
    task = get_task_cache_stats()
    embeddings = get_embedding_cache_stats()
    return {
        "task_response_cache": task,
        "embedding_cache": embeddings,
        "entries": task["entries"] + embeddings["entries"],
        "approximate_bytes": task["bytes"] + embeddings["approximate_bytes"],
        "persistent": False,
    }


def _cookbook_styles_snapshot():
    """List all cookbook styles with their variable requirements"""
    from pathlib import Path
    import json
    styles_dir = Path(__file__).parent / "presets" / "cookbook_styles"
    if not styles_dir.exists():
        return {"ok": True, "styles": [], "count": 0, "error": f"Styles dir not found: {styles_dir}"}
    
    styles = []
    for jf in sorted(styles_dir.glob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                env_vars = data.get("environment_variables", {})
                styles.append({
                    "slug": data.get("style_slug") or jf.stem,
                    "name": data.get("style_name") or jf.stem,
                    "display": f"{data.get('style_name', jf.stem)} ({data.get('style_slug', jf.stem)})",
                    "summary": data.get("style_summary", "")[:200],
                    "version": data.get("style_version", ""),
                    "variables": list(env_vars.keys()),
                    "variables_count": len(env_vars),
                    "variables_detail": env_vars,
                    "fidelity_anchors_count": len(data.get("style_fidelity_anchors", [])),
                    "file": jf.name,
                })
        except Exception as e:
            continue
    
    return {"ok": True, "styles": styles, "count": len(styles)}


def _cookbook_style_detail_snapshot(slug: str):
    """Get specific style.json detail"""
    from pathlib import Path
    import json
    styles_dir = Path(__file__).parent / "presets" / "cookbook_styles"
    # Try exact file
    for jf in styles_dir.glob("*.json"):
        if jf.stem == slug or slug in jf.stem or slug in jf.name:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {"ok": True, "style": data, "slug": slug, "file": jf.name}
            except Exception as e:
                return {"ok": False, "error": f"Failed to load {jf}: {e}"}
    # Try loading by display name
    try:
        with open(styles_dir / f"{slug}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"ok": True, "style": data, "slug": slug}
    except Exception as e:
        return {"ok": False, "error": f"Style not found: {slug} - {e}"}


async def _status_handler(_request):
    from aiohttp import web

    return web.json_response({"ok": True, **_STATUS, "cache": _cache_snapshot()})


async def _models_handler(request):
    from aiohttp import web

    try:
        base_url = dashboard_ollama_url(
            request.query.get("base_url", "http://127.0.0.1:11434")
        )
        result = await asyncio.to_thread(_ollama_snapshot, base_url)
        return web.json_response(result)
    except (ValueError, OllamaError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Ollama probe failed: {exc}"}, status=502)


async def _model_info_handler(request):
    from aiohttp import web

    try:
        base_url = dashboard_ollama_url(
            request.query.get("base_url", "http://127.0.0.1:11434")
        )
        model = request.query.get("model", "")
        result = await asyncio.to_thread(_model_capability_snapshot, base_url, model)
        return web.json_response(result)
    except (ValueError, OllamaError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Model inspection failed: {exc}"}, status=502)


async def _cache_handler(_request):
    from aiohttp import web

    return web.json_response({"ok": True, **_cache_snapshot()})


async def _cache_clear_handler(_request):
    from aiohttp import web

    cleared_tasks = clear_task_cache()
    cleared_embeddings = clear_embedding_cache()
    return web.json_response(
        {
            "ok": True,
            "cleared_entries": cleared_tasks + cleared_embeddings,
            **_cache_snapshot(),
        }
    )


async def _cookbook_styles_handler(_request):
    from aiohttp import web
    try:
        result = await asyncio.to_thread(_cookbook_styles_snapshot)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Cookbook styles list failed: {exc}"}, status=500)


async def _cookbook_style_detail_handler(request):
    from aiohttp import web
    try:
        slug = request.match_info.get("slug", "") or request.query.get("slug", "")
        if not slug:
            return web.json_response({"ok": False, "error": "slug required"}, status=400)
        result = await asyncio.to_thread(_cookbook_style_detail_snapshot, slug)
        if not result.get("ok"):
            return web.json_response(result, status=404)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Cookbook style detail failed: {exc}"}, status=500)


def register_routes(version: str, node_count: int, stability_tiers: dict | None = None) -> bool:
    """Register dashboard routes when imported by a running ComfyUI server."""

    global _ROUTES_REGISTERED
    _STATUS.update(
        version=version,
        node_count=int(node_count),
        stability_tiers=dict(stability_tiers or {}),
    )
    if _ROUTES_REGISTERED:
        return True
    try:
        from server import PromptServer
    except ImportError:
        return False
    if getattr(PromptServer, "instance", None) is None:
        return False
    routes = PromptServer.instance.routes
    routes.get("/comfy-omg/status")(_status_handler)
    routes.get("/comfy-omg/models")(_models_handler)
    routes.get("/comfy-omg/model-info")(_model_info_handler)
    routes.get("/comfy-omg/cache")(_cache_handler)
    routes.post("/comfy-omg/cache/clear")(_cache_clear_handler)
    routes.get("/comfy-omg/cookbook/styles")(_cookbook_styles_handler)
    routes.get("/comfy-omg/cookbook/style/{slug}")(_cookbook_style_detail_handler)
    routes.get("/comfy-omg/cookbook/style")(_cookbook_style_detail_handler)
    _ROUTES_REGISTERED = True
    return True
