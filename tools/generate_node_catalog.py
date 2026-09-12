"""Regenerate docs/NODES.md from the assembled ComfyUI-OMG mappings."""

from __future__ import annotations

import collections
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch  # noqa: F401
except ImportError:
    fake = types.ModuleType("torch")
    fake.Tensor = type("Tensor", (), {})
    fake.float32 = "float32"
    sys.modules["torch"] = fake

spec = importlib.util.spec_from_file_location(
    "comfy_omg_catalog", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not create ComfyUI-OMG package specification")
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

groups = collections.defaultdict(list)
for node_id, node_class in package.NODE_CLASS_MAPPINGS.items():
    groups[getattr(node_class, "CATEGORY", "Uncategorized")].append(
        (
            node_id,
            package.NODE_DISPLAY_NAME_MAPPINGS[node_id],
            getattr(node_class, "DESCRIPTION", ""),
        )
    )

lines = [
    "# ComfyUI-OMG Node Catalog",
    "",
    f"ComfyUI-OMG `{package.__version__}` currently registers "
    f"**{len(package.NODE_CLASS_MAPPINGS)} nodes**.",
    "",
    "The original 170 node IDs remain compatibility-sensitive; workflows store IDs even when "
    "display names change.",
    "",
]
for category in sorted(groups):
    lines.extend(
        [
            f"## {category}",
            "",
            "| Display name | Node ID | Purpose |",
            "|---|---|---|",
        ]
    )
    for node_id, display, description in sorted(groups[category], key=lambda item: item[1].lower()):
        purpose = " ".join(str(description).split()).replace("|", "\\|")
        lines.append(f"| {display} | `{node_id}` | {purpose} |")
    lines.append("")

output = ROOT / "docs" / "NODES.md"
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {output} with {len(package.NODE_CLASS_MAPPINGS)} nodes")
