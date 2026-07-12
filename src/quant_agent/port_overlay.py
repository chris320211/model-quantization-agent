"""Reviewable, content-addressed overlays for model-family ports.

The canonical method checkout is never edited. A port agent writes a unified diff
bundle under ``out/overlays``. Generated execution scripts must apply that diff to a
temporary detached Git worktree and run the method there.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import secrets
import shutil
from pathlib import Path, PurePosixPath

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .io_utils import atomic_write_text

_MAX_PATCH_BYTES = 200_000
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_OVERLAY_HEADER_RE = re.compile(r"^# QUANT_AGENT_OVERLAY_DIR=(.+)$", re.MULTILINE)


class PortOverlayManifest(BaseModel):
    schema_version: int = 1
    method_id: str
    model_id: str
    base_commit: str | None = None
    patch_sha256: str
    rationale: str = Field(..., min_length=1, max_length=4000)
    evidence_files: list[str] = Field(default_factory=list, max_length=20)
    target_modules: list[str] = Field(default_factory=list, max_length=200)


def _safe_patch_path(raw: str) -> bool:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if path.parts[0] in {".git", ".env"} or any(part.startswith(".env") for part in path.parts):
        return False
    return True


def validate_unified_patch(patch: str) -> list[str]:
    """Return patched repository-relative paths or raise on an unsafe/invalid diff."""
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("overlay patch must be non-empty")
    if len(patch.encode("utf-8")) > _MAX_PATCH_BYTES:
        raise ValueError(f"overlay patch exceeds {_MAX_PATCH_BYTES} bytes")
    if "GIT binary patch" in patch or "Binary files " in patch:
        raise ValueError("binary overlay patches are not supported")
    pairs = _DIFF_HEADER_RE.findall(patch)
    if not pairs:
        raise ValueError("overlay must be a git unified diff with diff --git headers")
    paths: list[str] = []
    for left, right in pairs:
        if left != right:
            raise ValueError("renames are not supported in port overlays")
        if not _safe_patch_path(left):
            raise ValueError(f"unsafe overlay path: {left!r}")
        paths.append(left)
    for line in patch.splitlines():
        if line.startswith(("--- ", "+++ ")):
            value = line[4:].split("\t", 1)[0]
            if value == "/dev/null":
                continue
            if not value.startswith(("a/", "b/")) or not _safe_patch_path(value[2:]):
                raise ValueError(f"unsafe unified diff path: {value!r}")
    return paths


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_overlay_bundle(path: Path) -> PortOverlayManifest:
    resolved = path.resolve()
    patch_path = resolved / "overlay.patch"
    manifest_path = resolved / "manifest.json"
    if not resolved.is_dir() or not patch_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"invalid port overlay bundle: {resolved}")
    manifest = PortOverlayManifest.model_validate_json(manifest_path.read_text())
    patch = patch_path.read_text()
    validate_unified_patch(patch)
    actual = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    if actual != manifest.patch_sha256:
        raise ValueError("port overlay patch hash does not match its manifest")
    return manifest


class PortOverlaySession:
    """Single-success overlay writer with validation and atomic file writes."""

    def __init__(
        self,
        *,
        root: Path,
        method_id: str,
        model_id: str,
        base_commit: str | None,
    ) -> None:
        self.root = root.resolve()
        self.method_id = method_id
        self.model_id = model_id
        self.base_commit = base_commit
        self.overlay_dir: Path | None = None
        self.manifest: PortOverlayManifest | None = None

    def write(
        self,
        *,
        patch: str,
        rationale: str,
        evidence_files: list[str] | None = None,
        target_modules: list[str] | None = None,
    ) -> dict:
        if self.overlay_dir is not None:
            return {"status": "error", "message": "port overlay is already finalized"}
        try:
            patched_paths = validate_unified_patch(patch)
            patch_sha = hashlib.sha256(patch.encode("utf-8")).hexdigest()
            manifest = PortOverlayManifest(
                method_id=self.method_id,
                model_id=self.model_id,
                base_commit=self.base_commit,
                patch_sha256=patch_sha,
                rationale=rationale,
                evidence_files=evidence_files or [],
                target_modules=target_modules or [],
            )
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        bundle_identity = "\0".join([
            self.method_id, self.model_id, self.base_commit or "unknown", patch_sha,
        ])
        bundle_sha = hashlib.sha256(bundle_identity.encode("utf-8")).hexdigest()
        final_dir = self.root / bundle_sha[:16]
        if not final_dir.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            temp_dir = self.root / f".{bundle_sha[:16]}.{secrets.token_hex(6)}.tmp"
            temp_dir.mkdir(mode=0o700)
            try:
                atomic_write_text(temp_dir / "overlay.patch", patch)
                atomic_write_text(temp_dir / "manifest.json", manifest.model_dump_json(indent=2))
                os.replace(temp_dir, final_dir)
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        stored_manifest = validate_overlay_bundle(final_dir)
        self.overlay_dir = final_dir
        self.manifest = stored_manifest
        return {
            "status": "ok",
            "overlay_dir": str(final_dir),
            "patch_sha256": patch_sha,
            "patched_paths": patched_paths,
            "manifest": stored_manifest.model_dump(),
        }


def make_write_port_overlay_tool(session: PortOverlaySession):
    @tool
    def write_port_overlay(
        patch: str,
        rationale: str,
        evidence_files: list[str] | None = None,
        target_modules: list[str] | None = None,
    ) -> str:
        """Validate and finalize a separate unified-diff port overlay.

        The diff must use repository-relative git headers, may add text files, and
        must not rename files, touch .git/.env, or contain binary patches.
        """
        return json.dumps(session.write(
            patch=patch,
            rationale=rationale,
            evidence_files=evidence_files,
            target_modules=target_modules,
        ), indent=2)

    return write_port_overlay


def overlay_path_from_script(code: str) -> Path | None:
    match = _OVERLAY_HEADER_RE.search(code)
    if not match:
        return None
    raw = match.group(1).strip()
    return Path(raw).expanduser().resolve()


def validate_overlay_script(code: str, overlay_dir: Path) -> None:
    """Require the generated wrapper contract that preserves the canonical checkout."""
    expected_header = f"# QUANT_AGENT_OVERLAY_DIR={overlay_dir.resolve()}"
    if expected_header not in code.splitlines()[:20]:
        raise ValueError("generated port script lacks the exact overlay header")
    validate_overlay_bundle(overlay_dir)
    tree = ast.parse(code)
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {"QUANT_AGENT_OVERLAY_DIR", "QUANT_AGENT_METHOD_REPO"}
    missing = sorted(required - literals)
    if missing:
        raise ValueError(
            "generated port script lacks executor-managed overlay environment usage: "
            + ", ".join(missing)
        )
