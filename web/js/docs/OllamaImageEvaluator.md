# Image Evaluator

## Purpose

Evaluates a candidate image against prompt, optional reference, and typed assets.

## Typical workflow

Image 1 is the candidate; Image 2 is the optional reference.

## Reliability and privacy

Critical issues force a fail verdict regardless of the model-reported score.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
