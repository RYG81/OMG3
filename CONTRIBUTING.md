# Contributing to ComfyUI-OMG

Thank you for helping improve ComfyUI-OMG.

## Development setup

1. Fork and clone the repository into `ComfyUI/custom_nodes/ComfyUI-OMG`.
2. Use ComfyUI's Python environment.
3. Install development tools: `python -m pip install -e '.[dev]'`.
4. Run `ruff check .` and `pytest` before opening a pull request.

## Pull requests

- Keep existing node IDs and socket order compatible unless a migration is included.
- Add tests for behavior changes.
- Do not perform network access from `INPUT_TYPES` or at module import time.
- Keep optional integrations lazy so the base pack still imports without their dependencies.
- Do not log private prompts, images, API keys, or full remote URLs containing credentials.
- Update `CHANGELOG.md` for user-visible changes.

## Adding a node

A new node should solve a workflow problem specific to ComfyUI-OMG's local creative-intelligence mission. Prefer a thin node over shared client/task infrastructure rather than copying request and JSON parsing code.

By contributing, you agree that your contribution is licensed under AGPL-3.0.
