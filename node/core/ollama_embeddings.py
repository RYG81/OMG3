"""
ollama_embeddings.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama Embeddings

Generates dense vector embeddings for text using /api/embed.
Useful for semantic search, clustering, RAG pipelines, etc.

The node outputs both a raw Python list (EMBEDDING) and a
cosine-similarity score if a second reference text is provided.

Inputs
  ollama_model    – OLLAMA_MODEL pipe (use an embed model, e.g. nomic-embed-text)
  text            – STRING  (primary text to embed)
  reference_text  – STRING  (optional; calculates cosine similarity)

Outputs
  EMBEDDING  – list[float]  (can feed into custom downstream nodes)
  FLOAT      – cosine similarity to reference_text  (-1 … 1)
  STRING     – JSON-serialised embedding vector (for logging / export)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import math

from ...utils.capabilities import require_declared_capability
from .ollama_client import embed, OllamaConnectionError


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match")
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class OllamaEmbeddings:
    DESCRIPTION = 'Ollama Embeddings'

    CATEGORY = "ComfyUI-OMG/Core"
    FUNCTION      = "embed_text"
    RETURN_TYPES  = ("EMBEDDING", "FLOAT", "STRING")
    RETURN_NAMES  = ("embedding", "cosine_similarity", "embedding_json")
    OUTPUT_NODE   = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL", {}),
                "text": (
                    "STRING",
                    {
                        "multiline":   True,
                        "default":     "Stable Diffusion is a latent diffusion model.",
                        "description": "Text to embed",
                    },
                ),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "reference_text": (
                    "STRING",
                    {
                        "multiline":   True,
                        "default":     "",
                        "description": "Optional: compute cosine similarity against this text",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default":     True,
                        "description": "L2-normalise the output vector",
                    },
                ),
                "truncate_dims": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 8192, "step": 1,
                        "description": "Truncate embedding to N dims (0 = keep all)",
                    },
                ),
            },
        }

    @staticmethod
    def _l2_normalize(v: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            return v
        return [x / norm for x in v]

    def embed_text(
        self,
        ollama_model:    dict,
        text:            str,
        reference_text:  str  = "",
        normalize:       bool = True,
        truncate_dims:   int  = 0,
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        if not text.strip():
            raise ValueError("text cannot be empty")
        require_declared_capability(ollama_model, "embeddings")
        base_url = ollama_model["base_url"]
        model    = ollama_model["model"]

        texts_to_embed = [text.strip()]
        if reference_text.strip():
            texts_to_embed.append(reference_text.strip())

        try:
            vectors = embed(base_url, model, texts_to_embed)
        except OllamaConnectionError as exc:
            raise RuntimeError(str(exc)) from exc

        if len(vectors) != len(texts_to_embed):
            raise RuntimeError(
                f"[OllamaEmbeddings] Expected {len(texts_to_embed)} vectors, received {len(vectors)}."
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) == 0:
            raise RuntimeError("[OllamaEmbeddings] Embedding dimensions are empty or inconsistent.")
        if not all(math.isfinite(float(value)) for vector in vectors for value in vector):
            raise RuntimeError("[OllamaEmbeddings] Embeddings contain non-finite values.")
        if any(sum(float(value) ** 2 for value in vector) == 0 for vector in vectors):
            raise RuntimeError("[OllamaEmbeddings] Embeddings contain a zero vector.")

        primary_vec = [float(value) for value in vectors[0]]

        # Optional truncation
        if truncate_dims > 0:
            primary_vec = primary_vec[:truncate_dims]

        # Optional normalisation
        if normalize:
            primary_vec = self._l2_normalize(primary_vec)

        # Cosine similarity
        sim = 0.0
        if len(vectors) >= 2:
            ref_vec = [float(value) for value in vectors[1]]
            if truncate_dims > 0:
                ref_vec = ref_vec[:truncate_dims]
            if normalize:
                ref_vec = self._l2_normalize(ref_vec)
            sim = _cosine_similarity(primary_vec, ref_vec)

        print(
            f"[OllamaEmbeddings] ✅ Dims: {len(primary_vec)} | "
            f"Cosine sim: {sim:.4f}"
        )

        embedding_json = json.dumps(primary_vec[:64]) + (
            f"… [{len(primary_vec)} dims total]" if len(primary_vec) > 64 else ""
        )

        return (primary_vec, sim, embedding_json)
