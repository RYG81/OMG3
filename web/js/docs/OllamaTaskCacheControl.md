# Ollama Task Cache Control

## Purpose

Inspects or clears task-response and embedding memory caches.

## Typical workflow

Use `stats` for counts/size or `clear` to evict all process-memory entries.

## Reliability and privacy

Caches are never persisted by this node. Saved RAG indexes are separate explicit files.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
