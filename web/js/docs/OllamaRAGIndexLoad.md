# RAG Index Load

## Purpose

Loads and validates an index from ComfyUI input or output.

## Typical workflow

Select the correct location and relative filename.

## Reliability and privacy

A 256 MB cap and vector validation apply.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
