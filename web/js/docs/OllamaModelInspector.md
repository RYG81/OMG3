# Ollama Model Inspector

## Purpose

Reads live model metadata and conservative capability flags.

## Typical workflow

Connect an `OLLAMA_MODEL` profile to inspect vision, tool, thinking, embedding, context, quantization, and loaded state.

## Reliability and privacy

Capability inference is conservative; declared server capabilities take precedence.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
