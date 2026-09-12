"""Local document chunking, embedding cache, vector index, and semantic retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_documents(documents: list[dict[str, Any]], source_kind: str) -> dict[str, Any]:
    normalized = []
    for index, item in enumerate(documents):
        text = str(item.get("text", ""))
        if not text.strip():
            continue
        source = str(item.get("source", f"document-{index + 1}"))
        metadata = item.get("metadata", {})
        normalized.append(
            {
                "document_id": _sha256(f"{source}\n{text}")[:20],
                "source": source,
                "text": text,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )
    return {
        "schema": "omg.documents",
        "version": 1,
        "valid": True,
        "source_kind": source_kind,
        "documents": normalized,
    }


def require_documents(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(envelope, dict) or envelope.get("schema") != "omg.documents":
        raise ValueError("Input is not an OMG Documents envelope")
    if envelope.get("valid") is False:
        raise ValueError("Documents envelope is marked invalid")
    documents = envelope.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Documents envelope contains no documents")
    return [item for item in documents if isinstance(item, dict)]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Split text at nearby whitespace while preserving deterministic offsets."""

    size = max(128, int(chunk_size))
    overlap = max(0, min(int(overlap), size - 1))
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        hard_end = min(length, start + size)
        end = hard_end
        if hard_end < length:
            boundary = max(text.rfind("\n", start + size // 2, hard_end), text.rfind(" ", start + size // 2, hard_end))
            if boundary > start:
                end = boundary
        value = text[start:end].strip()
        if value:
            chunks.append((start, end, value))
        if end >= length:
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return chunks


def build_chunks(
    documents: list[dict[str, Any]], chunk_size: int, overlap: int, max_chunks: int
) -> list[dict[str, Any]]:
    chunks = []
    for document in documents:
        source = str(document.get("source", "document"))
        doc_id = str(document.get("document_id", _sha256(source)[:20]))
        metadata = document.get("metadata", {})
        for index, (start, end, text) in enumerate(chunk_text(str(document.get("text", "")), chunk_size, overlap)):
            chunk_id = _sha256(f"{doc_id}\n{index}\n{text}")[:24]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "source": source,
                    "chunk_index": index,
                    "start": start,
                    "end": end,
                    "text": text,
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }
            )
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def normalize_vector(vector: list[float]) -> list[float]:
    values = [float(value) for value in vector]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding vector is empty or contains non-finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("Embedding vector has zero norm")
    return [value / norm for value in values]


class _EmbeddingCache:
    def __init__(self, max_entries: int = 4096):
        self.max_entries = max_entries
        self._items: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
                return list(value)
            return None

    def put(self, key: str, value: list[float]) -> None:
        with self._lock:
            self._items[key] = list(value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._items),
                "approximate_bytes": sum(len(vector) * 8 for vector in self._items.values()),
                "max_entries": self.max_entries,
            }


EMBEDDING_CACHE = _EmbeddingCache()


def clear_embedding_cache() -> int:
    return EMBEDDING_CACHE.clear()


def get_embedding_cache_stats() -> dict[str, int]:
    return EMBEDDING_CACHE.stats()


def _embedding_key(model_profile: dict, text: str) -> str:
    return _sha256(
        json.dumps(
            {
                "endpoint": model_profile.get("base_url", ""),
                "model": model_profile.get("model", ""),
                "text_sha256": _sha256(text),
            },
            sort_keys=True,
        )
    )


def embed_texts_cached(
    model_profile: dict,
    texts: list[str],
    batch_size: int,
    cache_policy: str,
    embed_fn: Callable[..., list[list[float]]],
) -> tuple[list[list[float]], int, int]:
    if cache_policy not in {"use", "refresh", "bypass"}:
        raise ValueError("cache_policy must be use, refresh, or bypass")
    vectors: list[list[float] | None] = [None] * len(texts)
    misses = []
    hit_count = 0
    for index, text in enumerate(texts):
        key = _embedding_key(model_profile, text)
        cached = EMBEDDING_CACHE.get(key) if cache_policy == "use" else None
        if cached is None:
            misses.append((index, text, key))
        else:
            vectors[index] = cached
            hit_count += 1

    size = max(1, int(batch_size))
    for offset in range(0, len(misses), size):
        batch = misses[offset : offset + size]
        batch_texts = [item[1] for item in batch]
        generated = embed_fn(model_profile["base_url"], model_profile["model"], batch_texts)
        if len(generated) != len(batch):
            raise ValueError("Ollama returned a different number of embeddings than requested")
        for (index, _text, key), vector in zip(batch, generated):
            normalized = normalize_vector(vector)
            vectors[index] = normalized
            if cache_policy != "bypass":
                EMBEDDING_CACHE.put(key, normalized)

    result = [vector for vector in vectors if vector is not None]
    if len(result) != len(texts):
        raise ValueError("Failed to obtain every requested embedding")
    dimensions = {len(vector) for vector in result}
    if len(dimensions) != 1:
        raise ValueError("Embedding dimensions are inconsistent")
    return result, hit_count, len(misses)


def build_index(
    model_profile: dict,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    chunk_size: int,
    overlap: int,
    cache_hits: int,
) -> dict[str, Any]:
    if len(chunks) != len(vectors):
        raise ValueError("Chunk and vector counts differ")
    dimensions = len(vectors[0]) if vectors else 0
    indexed_chunks = []
    for chunk, vector in zip(chunks, vectors):
        indexed_chunks.append({**chunk, "vector": vector})
    fingerprint = _sha256(
        json.dumps(
            {
                "model": model_profile.get("model", ""),
                "chunks": [chunk["chunk_id"] for chunk in chunks],
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
            sort_keys=True,
        )
    )
    return {
        "schema": "omg.rag_index",
        "version": 1,
        "valid": True,
        "index_id": fingerprint[:24],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": {
            "model": model_profile.get("model", ""),
            "base_url": model_profile.get("base_url", ""),
            "dimensions": dimensions,
        },
        "chunking": {"chunk_size": chunk_size, "overlap": overlap},
        "cache_hits": cache_hits,
        "chunks": indexed_chunks,
    }


def require_index(index: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(index, dict) or index.get("schema") != "omg.rag_index":
        raise ValueError("Input is not an OMG RAG Index")
    if index.get("valid") is False:
        raise ValueError("RAG Index is marked invalid")
    chunks = index.get("chunks")
    model = index.get("embedding_model")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("RAG Index contains no chunks")
    if len(chunks) > 10000:
        raise ValueError("RAG Index exceeds the 10,000 chunk safety limit")
    if not isinstance(model, dict) or int(model.get("dimensions", 0)) <= 0:
        raise ValueError("RAG Index embedding metadata is invalid")
    dimensions = int(model["dimensions"])
    for chunk in chunks:
        vector = chunk.get("vector") if isinstance(chunk, dict) else None
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError("RAG Index contains an invalid vector")
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("RAG Index contains a non-finite vector")
    return index


def cosine_similarity(normalized_a: list[float], normalized_b: list[float]) -> float:
    if len(normalized_a) != len(normalized_b):
        raise ValueError("Embedding dimensions do not match")
    return sum(a * b for a, b in zip(normalized_a, normalized_b))


def search_index(index: dict[str, Any], query_vector: list[float], top_k: int, min_score: float):
    validated = require_index(index)
    normalized_query = normalize_vector(query_vector)
    scored = []
    for chunk in validated["chunks"]:
        score = cosine_similarity(normalized_query, [float(value) for value in chunk["vector"]])
        if score >= min_score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: max(1, int(top_k))]


def index_summary(index: dict[str, Any]) -> dict[str, Any]:
    validated = require_index(index)
    sources = sorted({str(chunk.get("source", "")) for chunk in validated["chunks"]})
    return {
        "schema": validated["schema"],
        "version": validated["version"],
        "index_id": validated["index_id"],
        "created_at_utc": validated.get("created_at_utc", ""),
        "embedding_model": validated["embedding_model"],
        "chunking": validated["chunking"],
        "chunk_count": len(validated["chunks"]),
        "source_count": len(sources),
        "sources": sources,
        "vectors_omitted": True,
    }
