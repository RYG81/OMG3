# RAG Semantic Search

## Purpose

Retrieves top local chunks with cosine similarity and citations.

## Typical workflow

Use the same embedding model used to build the index. Connect `context` to a prompt/agent node.

## Reliability and privacy

Query/index model mismatch is rejected.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
