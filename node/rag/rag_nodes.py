"""Local document loading, embedding index construction, retrieval, and persistence nodes."""

from __future__ import annotations

import json
import os

from ...ollama_client import embed
from ...tasks.rag import (
    build_chunks,
    build_index,
    embed_texts_cached,
    index_summary,
    make_documents,
    require_documents,
    require_index,
    search_index,
)
from ...utils.capabilities import require_declared_capability
from ...utils.path_utils import ensure_within, resolve_input_path, resolve_output_path, safe_join
from ...utils.progress import ProgressReporter

_ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".yaml", ".yml"}


class OllamaDocumentsFromText:
    """Create a typed local document collection from a text input."""

    CATEGORY = "ComfyUI-OMG/RAG"
    FUNCTION = "create_documents"
    RETURN_TYPES = ("OMG_DOCUMENTS", "STRING", "INT", "INT")
    RETURN_NAMES = ("documents", "documents_json", "document_count", "character_count")
    DESCRIPTION = "Create a typed OMG Documents collection from text without filesystem access."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "source_name": ("STRING", {"default": "manual-input"}),
            }
        }

    def create_documents(self, text: str, source_name: str):
        if not text.strip():
            raise ValueError("text cannot be empty")
        envelope = make_documents(
            [{"source": source_name.strip() or "manual-input", "text": text, "metadata": {}}],
            "manual_text",
        )
        return (
            envelope,
            json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True),
            len(envelope["documents"]),
            len(text),
        )


class OllamaReferenceFolderLoader:
    """Load bounded text references from a ComfyUI input folder."""

    CATEGORY = "ComfyUI-OMG/RAG"
    FUNCTION = "load_folder"
    RETURN_TYPES = ("OMG_DOCUMENTS", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "documents",
        "documents_json",
        "document_count",
        "total_characters",
        "load_report",
    )
    DESCRIPTION = (
        "Load text, Markdown, JSON, CSV, or YAML references from ComfyUI input with file-count "
        "and byte safety limits."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "references"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "recursive": ("BOOLEAN", {"default": True}),
                "max_files": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "max_total_megabytes": (
                    "FLOAT",
                    {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0},
                ),
                "allow_external_paths": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def load_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        max_files: int = 100,
        max_total_megabytes: float = 20.0,
        allow_external_paths: bool = False,
    ):
        folder = resolve_input_path(folder_path, allow_external=allow_external_paths)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Reference folder not found: {folder}")
        pattern = "**/*" if recursive else "*"
        candidates = sorted(
            path
            for path in folder.glob(pattern)
            if path.is_file() and path.suffix.lower() in _ALLOWED_EXTENSIONS
        )
        documents = []
        errors = []
        total_bytes = 0
        max_total_bytes = int(max_total_megabytes * 1024 * 1024)
        progress = ProgressReporter(min(len(candidates), max_files) or 1)
        for path in candidates[:max_files]:
            progress.check_interrupted()
            try:
                resolved = ensure_within(folder, path)
                size = resolved.stat().st_size
                if size > 2 * 1024 * 1024:
                    raise ValueError("file exceeds 2 MB per-file limit")
                if total_bytes + size > max_total_bytes:
                    errors.append("Stopped at total byte limit")
                    break
                text = resolved.read_text(encoding="utf-8")
                total_bytes += size
                documents.append(
                    {
                        "source": str(resolved.relative_to(folder)),
                        "text": text,
                        "metadata": {"extension": resolved.suffix.lower(), "bytes": size},
                    }
                )
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")
            progress.update()
        envelope = make_documents(documents, "reference_folder")
        if not envelope["documents"]:
            raise ValueError("No readable reference documents were loaded")
        report = {
            "folder": str(folder),
            "loaded": len(envelope["documents"]),
            "bytes": total_bytes,
            "errors": errors,
        }
        return (
            envelope,
            json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True),
            len(envelope["documents"]),
            sum(len(item["text"]) for item in envelope["documents"]),
            json.dumps(report, indent=2, ensure_ascii=False),
        )


class OllamaRAGIndexBuild:
    """Chunk local documents and embed them into a typed in-memory vector index."""

    CATEGORY = "ComfyUI-OMG/RAG"
    FUNCTION = "build_rag_index"
    RETURN_TYPES = ("OMG_RAG_INDEX", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("rag_index", "index_summary_json", "chunk_count", "dimensions", "sources")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Build a local semantic index with Ollama embeddings, deterministic chunks, normalized "
        "vectors, and a bounded process-memory embedding cache."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "documents": ("OMG_DOCUMENTS", {"forceInput": True}),
            },
            "optional": {
                "chunk_size": ("INT", {"default": 1200, "min": 128, "max": 8000}),
                "chunk_overlap": ("INT", {"default": 150, "min": 0, "max": 2000}),
                "max_chunks": ("INT", {"default": 256, "min": 1, "max": 2000}),
                "embedding_batch_size": ("INT", {"default": 16, "min": 1, "max": 128}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def build_rag_index(
        self,
        ollama_model: dict,
        documents: dict,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        max_chunks: int = 256,
        embedding_batch_size: int = 16,
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        require_declared_capability(ollama_model, "embeddings")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        docs = require_documents(documents)
        chunks = build_chunks(docs, chunk_size, chunk_overlap, max_chunks)
        if not chunks:
            raise ValueError("No non-empty chunks were produced")
        progress = ProgressReporter(2)
        progress.check_interrupted()
        vectors, cache_hits, _misses = embed_texts_cached(
            ollama_model,
            [chunk["text"] for chunk in chunks],
            embedding_batch_size,
            cache_policy,
            embed,
        )
        progress.update()
        index = build_index(
            ollama_model, chunks, vectors, chunk_size, chunk_overlap, cache_hits
        )
        summary = index_summary(index)
        progress.update()
        return (
            index,
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            summary["chunk_count"],
            summary["embedding_model"]["dimensions"],
            "\n".join(summary["sources"]),
        )


class OllamaRAGSearch:
    """Embed a query and retrieve the most similar local reference chunks."""

    CATEGORY = "ComfyUI-OMG/RAG"
    FUNCTION = "search"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("context", "citations_json", "scores", "result_count", "top_source")
    DESCRIPTION = "Retrieve top local reference chunks with cosine similarity and source citations."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "rag_index": ("OMG_RAG_INDEX", {"forceInput": True}),
                "query": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "top_k": ("INT", {"default": 5, "min": 1, "max": 50}),
                "minimum_score": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "max_context_characters": (
                    "INT",
                    {"default": 12000, "min": 500, "max": 100000},
                ),
            },
        }

    def search(
        self,
        ollama_model: dict,
        rag_index: dict,
        query: str,
        top_k: int = 5,
        minimum_score: float = 0.0,
        max_context_characters: int = 12000,
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        if not query.strip():
            raise ValueError("query cannot be empty")
        index = require_index(rag_index)
        expected_model = str(index["embedding_model"].get("model", ""))
        if ollama_model.get("model") != expected_model:
            raise ValueError(
                f"Query model {ollama_model.get('model')!r} does not match index model {expected_model!r}"
            )
        require_declared_capability(ollama_model, "embeddings")
        vectors, _hits, _misses = embed_texts_cached(
            ollama_model, [query], 1, "use", embed
        )
        results = search_index(index, vectors[0], top_k, minimum_score)
        citations = []
        context_parts = []
        current_length = 0
        for rank, (score, chunk) in enumerate(results, start=1):
            header = f"[{rank}] {chunk.get('source', '')}#chunk-{chunk.get('chunk_index', 0)}"
            block = f"{header}\n{chunk.get('text', '')}"
            if context_parts and current_length + len(block) > max_context_characters:
                break
            context_parts.append(block)
            current_length += len(block)
            citations.append(
                {
                    "rank": rank,
                    "score": score,
                    "source": chunk.get("source", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "start": chunk.get("start", 0),
                    "end": chunk.get("end", 0),
                }
            )
        return (
            "\n\n".join(context_parts),
            json.dumps(citations, indent=2, ensure_ascii=False),
            ", ".join(f"{item['score']:.4f}" for item in citations),
            len(citations),
            str(citations[0]["source"]) if citations else "",
        )


class OllamaRAGIndexSave:
    """Persist a validated local RAG index inside ComfyUI output."""

    CATEGORY = "ComfyUI-OMG/RAG"
    FUNCTION = "save_index"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("filepath", "size_bytes", "summary_json")
    OUTPUT_NODE = True
    DESCRIPTION = "Save a validated RAG index, including local text and vectors, under ComfyUI output."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rag_index": ("OMG_RAG_INDEX", {"forceInput": True}),
                "filename": ("STRING", {"default": "comfy_omg/rag_index.json"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    def save_index(self, rag_index: dict, filename: str, overwrite: bool):
        index = require_index(rag_index)
        path = resolve_output_path(filename)
        if path.suffix.lower() != ".json":
            raise ValueError("RAG index filename must end in .json")
        if path.exists() and not overwrite:
            raise FileExistsError(f"RAG index already exists: {path}")
        payload = json.dumps(index, ensure_ascii=False, separators=(",", ":"))
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, path)
        return (
            str(path),
            path.stat().st_size,
            json.dumps(index_summary(index), indent=2, ensure_ascii=False, sort_keys=True),
        )


class OllamaRAGIndexLoad:
    """Load and validate a persisted RAG index from ComfyUI input or output."""

    CATEGORY = "ComfyUI-OMG/RAG"
    FUNCTION = "load_index"
    RETURN_TYPES = ("OMG_RAG_INDEX", "STRING", "INT", "INT")
    RETURN_NAMES = ("rag_index", "summary_json", "chunk_count", "dimensions")
    DESCRIPTION = "Load a validated local RAG index from ComfyUI input or output."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "comfy_omg/rag_index.json"}),
                "location": (["input", "output"], {"default": "input"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def load_index(self, filename: str, location: str):
        if location == "input":
            path = resolve_input_path(filename)
        else:
            import folder_paths

            path = safe_join(folder_paths.get_output_directory(), filename)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"RAG index not found: {path}")
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError("RAG index exceeds the 256 MB load limit")
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"RAG index JSON is invalid: {exc}") from exc
        index = require_index(index)
        summary = index_summary(index)
        return (
            index,
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            summary["chunk_count"],
            summary["embedding_model"]["dimensions"],
        )
