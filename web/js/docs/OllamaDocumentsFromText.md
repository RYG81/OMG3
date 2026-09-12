# RAG Documents from Text

## Purpose

Creates a typed `OMG_DOCUMENTS` collection from manual text.

## Typical workflow

Provide a stable source name for meaningful citations.

## Reliability and privacy

No filesystem or Ollama call occurs.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
