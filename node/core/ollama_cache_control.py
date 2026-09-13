"""Inspect or clear ComfyUI-OMG's private process-memory response cache."""

from __future__ import annotations

import json

from ...tasks.engine import clear_task_cache, get_task_cache_stats
from ...tasks.rag import clear_embedding_cache, get_embedding_cache_stats


class OllamaTaskCacheControl:
    """Inspect or clear the bounded in-memory cache used by schema-driven tasks."""

    CATEGORY = "ComfyUI-OMG/Core"
    FUNCTION = "manage_cache"
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("cache_report_json", "entries", "size_megabytes")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Inspect or clear the private, bounded, process-memory cache used by ComfyUI-OMG tasks. "
        "Cached prompts and responses are never written to disk."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["stats", "clear"], {"default": "stats"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def manage_cache(self, action: str):
        cleared_tasks = clear_task_cache() if action == "clear" else 0
        cleared_embeddings = clear_embedding_cache() if action == "clear" else 0
        task_stats = get_task_cache_stats()
        embedding_stats = get_embedding_cache_stats()
        total_entries = task_stats["entries"] + embedding_stats["entries"]
        total_bytes = task_stats["bytes"] + embedding_stats["approximate_bytes"]
        report = {
            "action": action,
            "cleared_entries": cleared_tasks + cleared_embeddings,
            "task_response_cache": task_stats,
            "embedding_cache": embedding_stats,
            "entries": total_entries,
            "bytes": total_bytes,
            "persistent": False,
        }
        return (
            json.dumps(report, indent=2, sort_keys=True),
            total_entries,
            total_bytes / (1024 * 1024),
        )
