# RAG Index Build

## Purpose

Chunks documents and embeds them with an Ollama embedding model.

## Typical workflow

Use an embedding model such as `nomic-embed-text`; tune chunk size/overlap for the source material.

## Reliability and privacy

Vectors are normalized and dimension-checked. The summary omits vectors.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
