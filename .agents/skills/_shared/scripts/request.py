"""Request JSON handoff between skills. No secrets."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import atomic_write_text
from paths import REQUESTS_ROOT, require_slug

_SLUGIFY_RE = re.compile(r"[^a-z0-9]+")
_SECRET_KEY_RE = re.compile(
    r"token|secret|password|api[_-]?key|hf_token|openai|github",
    re.I,
)

REQUIRED = (
    "method_name",
    "model_id",
    "gpu_instance",
    "arxiv_id",
    "paper_path",
    "repo_url",
    "repo_path",
    "repo_commit",
    "hf_snapshot_path",
    "slug",
)

OPTIONAL = (
    "instance_type",
    "gpu_arch",
    "vram_gb",
    "gpu",
    "gpu_count",
    "compute_capability",
    "gpu_name",
    "nvidia_smi",
    "gpus",
    "vram_gb_free",
    "vram_gb_total_live",
    "venv_python",
    "winner_overlay_dir",
    "winner_script",
    "ranked",
)

ALLOWED_KEYS = frozenset(REQUIRED + OPTIONAL)


def slugify(*parts: str) -> str:
    raw = "-".join(parts)
    return _SLUGIFY_RE.sub("-", raw.lower()).strip("-")[:80] or "request"


def request_path(slug: str) -> Path:
    require_slug(slug)
    return REQUESTS_ROOT / f"{slug}.json"


def _reject_secret_keys(data: dict) -> None:
    for key in data:
        if _SECRET_KEY_RE.search(str(key)):
            raise ValueError(f"request JSON must not contain secret-like key {key!r}")


def _validate(data: dict) -> dict:
    _reject_secret_keys(data)
    unknown = sorted(set(data) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"request JSON has unsupported keys: {unknown}")
    missing = [key for key in REQUIRED if not data.get(key)]
    if missing:
        raise ValueError(f"request JSON missing {missing}")
    require_slug(str(data["slug"]))
    return data


def load_request(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("request JSON must be an object")
    return _validate(data)


def write_request(payload: dict) -> Path:
    if not isinstance(payload, dict):
        raise ValueError("request JSON must be an object")
    _reject_secret_keys(payload)
    slug = payload.get("slug") or slugify(
        str(payload.get("method_name", "method")),
        str(payload.get("model_id", "model")),
        str(payload.get("gpu_instance", "gpu")),
    )
    require_slug(str(slug))
    filtered = {key: payload[key] for key in ALLOWED_KEYS if key in payload}
    filtered["slug"] = slug
    _validate(filtered)
    path = request_path(slug)
    atomic_write_text(path, json.dumps(filtered, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Read or write a gather request JSON")
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--write-json", help="JSON object to write")
    args = parser.parse_args()
    if args.write_json:
        path = write_request(json.loads(args.write_json))
        print(path)
        return 0
    if args.path is None:
        raise SystemExit("path required")
    print(json.dumps(load_request(args.path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
