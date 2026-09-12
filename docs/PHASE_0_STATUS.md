# Phase 0 Status

This document records the stabilization work on branch `phase-0-foundation` before the first public beta.

## Completed

- [x] Preserve and contract-test all 170 existing node IDs.
- [x] Rebrand all user-facing category roots as `ComfyUI-OMG` without renaming serialized node IDs.
- [x] Add AGPL-3.0 licensing and copyright notice.
- [x] Replace placeholder README with installation, quick-start, optional-dependency, privacy, and troubleshooting documentation.
- [x] Add a generated catalog covering all current nodes.
- [x] Add `pyproject.toml` with `comfy-omg` / publisher `RYG81` Registry metadata.
- [x] Add changelog, contributing guide, security policy, issue template, and pull-request template.
- [x] Remove 90 tracked Python bytecode/cache files and add `.gitignore` / `.comfyignore`.
- [x] Remove the subprocess-based pip installer.
- [x] Consolidate the duplicate Ollama clients into one canonical implementation.
- [x] Consolidate the duplicate model loaders.
- [x] Make Model Loader schema discovery offline-safe and expose `num_ctx`.
- [x] Support local and remote HTTP(S) Ollama URLs without embedding credentials.
- [x] Add one normalized Ollama error hierarchy and strict stream parsing.
- [x] Add JSON Schema objects as supported Ollama response formats.
- [x] Remove incomplete `WEB_DIRECTORY` declaration and inaccurate frontend claims.
- [x] Restore complete default ComfyUI fingerprints for both Prompt Builder nodes.
- [x] Add a non-empty `DESCRIPTION` to every registered node.
- [x] Sandboxed text and video files to ComfyUI input/output directories.
- [x] Require explicit opt-in for external photoset folders.
- [x] Replace expression `eval` with a bounded AST-whitelist evaluator.
- [x] Enable web SSL verification by default.
- [x] Add public-address validation, redirect checks, download-size limits, and image-pixel limits.
- [x] Add optional progress/cancellation integration for long-running operations.
- [x] Reduce base dependencies and move OpenCV / smolagents integrations to optional extras.
- [x] Resolve all default Ruff diagnostics.
- [x] Add offline tests for mappings, schemas, clients, paths, network validation, and expressions.
- [x] Add Python 3.10–3.13 CI, high-severity Bandit scan, source-package build, and manual Registry publishing workflow.

## Current automated result

- `ruff check .`: passing
- `python -m compileall -q .`: passing
- `pytest`: 137 passing, including a real local HTTP mock of the Ollama API and Model Loader
- registered nodes: 204 (all original 170 plus thirty-four additive nodes)
- matching display mappings: 204
- missing node descriptions: 0
- offline import/schema contract: passing
- source distribution build: passing
- high-severity Bandit findings: 0
- known vulnerabilities in mandatory requirements (`pip-audit`): 0

## Remaining before publishing `0.1.0`

The owner chose not to block the alpha release on cross-platform/manual runtime testing. Automated contract and mock-Ollama coverage remains mandatory. Real ComfyUI/Ollama and Windows/macOS smoke tests are deferred until user feedback or a release candidate.

- [ ] Add at least three starter workflows with thumbnails.
- [ ] Confirm that the Comfy Registry publisher ID is exactly `RYG81` and create `REGISTRY_ACCESS_TOKEN`.
- [ ] Decide whether `comfy-omg` should be published now; Registry node names are effectively long-lived identifiers.
- [ ] Set the `0.1.0` date, update the changelog, tag the commit, and manually run the publish workflow.

## Deferred, non-blocking validation

- [ ] Test the branch in a real ComfyUI installation with real torch tensors.
- [ ] Test text, chat, vision, embedding, model-pull, and web workflows against a real Ollama server.
- [ ] Validate Windows Portable, Linux, and macOS installations.

Until those checks happen, releases should retain an **alpha/beta** label rather than claim production-stable cross-platform support.

## Compatibility note

No existing node ID was removed. The Model Loader keeps its historical node ID and outputs, but model selection is now an offline-safe text field instead of a localhost-derived combo. Existing model-name widget values remain strings. A new optional `num_ctx` field was appended.
