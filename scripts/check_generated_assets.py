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
FILES = (
    "methods.yaml",
    "method_capabilities.yaml",
    "model_aliases.yaml",
    "aws_instances.yaml",
    "gpu_specs.yaml",
)

MIRRORED_SKILL_FILES = (
    "quant/SKILL.md",
    "quant/reference/pipeline_contract.md",
    "quant/scripts/evaluate_compatibility.py",
    "quant/scripts/method_env.py",
    "quant/scripts/write_overlay.py",
    "quant/scripts/validate_script.py",
    "quant-execute/SKILL.md",
    "quant-execute/scripts/launch_job.py",
    "quant-tune/SKILL.md",
    "quant-setup/SKILL.md",
)


def main() -> int:
    errors: list[str] = []
    for mirror in MIRRORS:
        for name in FILES:
            if (CANON / name).read_bytes() != (mirror / name).read_bytes():
                errors.append(f"catalog drift: {mirror / name}")

    agent_skills = ROOT / ".agents" / "skills"
    claude_skills = ROOT / ".claude" / "skills"
    for relative in MIRRORED_SKILL_FILES:
        if (agent_skills / relative).read_bytes() != (claude_skills / relative).read_bytes():
            errors.append(f"skill mirror drift: {relative}")

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
        for obsolete in (
            "/tmp/quant-",
            "does not build venvs",
            "AST-only validation",
            "stage 1 (`ast.parse`) only",
        ):
            if obsolete in text:
                errors.append(f"obsolete skill pipeline text {obsolete!r}: {path}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("generated assets are synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
