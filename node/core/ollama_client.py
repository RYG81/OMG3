"""Compatibility import for the canonical ComfyUI-OMG Ollama client.

Core nodes historically imported ``nodes.core.ollama_client`` while creative
nodes imported the package-root client. Keeping this thin module preserves those
imports while ensuring every node shares one implementation, error hierarchy,
and HTTP session cache.
"""

from ...ollama_client import *  # noqa: F403
