# ComfyUI-OMG Local Loaders

The `ComfyUI-OMG/Loaders` category provides local-only, dependency-free input nodes. They do not contact Ollama or any network service.

## Image From Folder

Loads one clamped, 1-based item from a naturally sorted image folder. Only the selected image is decoded, making this the preferred node for iterating through large folders with low memory use. It returns the image, matching alpha mask, current index, available count, workflow-safe path, and manifest.

## Image Folder Batch Loader

Loads supported images from a folder under ComfyUI input and returns:

1. normalized `IMAGE` batch;
2. matching `MASK` batch (`1` means transparent/masked, matching ComfyUI LoadImage semantics);
3. loaded image count;
4. workflow-safe newline paths;
5. a versioned JSON manifest.

Files are naturally sorted (`frame2` before `frame10`). Recursive search, filename filtering, start index, stride, and maximum image count are available. `contain_pad`, `cover_crop`, and `stretch` always produce a stackable batch with one output size.

## Multiple Image Paths Loader

Accepts ordered paths as either:

- one path per line; or
- a JSON array of strings.

The path list from Folder File Catalog connects directly to this node. Input order is retained rather than sorted.

## Folder File Catalog

Scans metadata only; file contents and media are not decoded. Families:

- images: PNG, JPEG, WebP, BMP, TIFF;
- video: MP4, MOV, MKV, WebM, AVI, M4V;
- audio: WAV, MP3, FLAC, OGG, Opus, M4A, AAC;
- text/data: text, Markdown, JSON/JSONL, CSV/TSV, YAML, XML/HTML, SRT/VTT, logs, prompt, INI, and CFG.

Outputs include feedable paths, a versioned manifest, count, total bytes, and truncation status.

## Text & Data File Loader

Always exposes decoded source text. In `auto` or `normalized` mode it also produces normalized JSON for:

- JSON;
- JSON Lines;
- CSV;
- TSV.

YAML, XML, HTML, subtitles, configuration, prompt, and normal text files remain raw text so the base package does not need additional parsers or dependencies.

## Safety and performance limits

- Paths are restricted to ComfyUI input by default.
- External absolute paths require explicit opt-in.
- Symlinks must remain inside the selected folder.
- Folder scans stop after 10,000 entries.
- Image source files are capped at 100 MB and 40 megapixels each.
- A complete output batch is capped at 4,194,304 pixels (about 64 MB for RGB plus mask float32 tensors).
- Direct path lists are capped at 256 entries; image outputs are capped at 64 and may be reduced further by the total-pixel limit.
- Text/data files default to 5 MB and cannot exceed a configured 20 MB.
- JSONL/CSV/TSV row output is bounded.
- File fingerprints use path, size, and modification time so unchanged loader nodes can reuse ComfyUI execution results.

Manifests report truncation, skipped files, errors, active limits, and loaded dimensions instead of silently claiming a complete load.
