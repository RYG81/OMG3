# RAG Index Save

## Purpose

Atomically saves an index under ComfyUI output.

## Typical workflow

Use a `.json` filename and enable overwrite only deliberately.

## Reliability and privacy

Saved files contain source text and embedding vectors.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
