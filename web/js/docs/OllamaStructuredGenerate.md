# Ollama Structured Generate

## Purpose

Generates JSON constrained by a Draft 2020-12 JSON Schema.

## Typical workflow

Supply a prompt and schema. Use `validated_json` only when `is_valid` is true; inspect the validation report and provenance when debugging.

## Reliability and privacy

One optional repair attempt is supported. `use`, `refresh`, and `bypass` control the private memory cache.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
