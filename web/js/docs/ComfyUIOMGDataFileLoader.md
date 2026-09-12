# Text & Data File Loader

## Purpose

Loads bounded text, prompt, subtitle, and structured-data files from ComfyUI input.

## Typical workflow

Use raw text directly, or consume normalized JSON for JSON, JSONL, CSV, and TSV files.

## Reliability and privacy

No network or LLM call occurs. Extensions, byte size, encoding, and parsed row counts are restricted.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
