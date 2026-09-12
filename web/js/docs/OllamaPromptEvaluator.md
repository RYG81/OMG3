# Prompt Evaluator

## Purpose

Evaluates a prompt against intent, model family, and optional typed assets.

## Typical workflow

Use normalized score/verdict and severity-ranked issues; feed `improved_prompt` into the next iteration.

## Reliability and privacy

The final weighted score and verdict are recomputed locally.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
