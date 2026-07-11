#!/usr/bin/env python3
"""Fail when packaged catalogs, skill mirrors, or documented paths drift."""
from __future__ import annotations

from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "src" / "quant_agent" / "data"
MIRRORS = [
    ROOT / ".agents" / "skills" / "quant" / "reference",
    ROOT / ".claude" / "skills" / "quant" / "reference",
]
FILES = ("methods.yaml", "model_aliases.yaml", "aws_instances.yaml", "gpu_specs.yaml")


def main() -> int:
    errors: list[str] = []
    for mirror in MIRRORS:
        for name in FILES:
            if (CANON / name).read_bytes() != (mirror / name).read_bytes():
                errors.append(f"catalog drift: {mirror / name}")

    methods = yaml.safe_load((CANON / "methods.yaml").read_text())
    if len(methods) != 35:
        errors.append(f"expected 35 methods, found {len(methods)}")

    capabilities = yaml.safe_load((CANON / "method_capabilities.yaml").read_text())
    capability_methods = (capabilities or {}).get("methods", {})
    catalog_ids = {row["id"] for row in methods}
    capability_ids = set(capability_methods)
    if capability_ids != catalog_ids:
        errors.append(
            "capability catalog drift: "
            f"missing={sorted(catalog_ids - capability_ids)}, "
            f"extra={sorted(capability_ids - catalog_ids)}"
        )

    text_files = [ROOT / "README.md", *ROOT.glob(".agents/skills/*/SKILL.md")]
    for path in text_files:
        text = path.read_text()
        if ".Codex/" in text:
            errors.append(f"obsolete .Codex path: {path}")
        if "34 methods" in text or "34-method" in text:
            errors.append(f"obsolete method count: {path}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("generated assets are synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
