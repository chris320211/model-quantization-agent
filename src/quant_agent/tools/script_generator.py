from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from langchain_core.tools import tool

from ..config import load_settings
from .recommender import load_catalog

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
        keep_trailing_newline=True,
    )


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def render(method_id: str, model_id: str, options: dict | None = None) -> str:
    catalog = {m["id"]: m for m in load_catalog()}
    if method_id not in catalog:
        raise ValueError(f"Unknown method id: {method_id}")
    tpl_name = catalog[method_id].get("template")
    if not tpl_name:
        raise ValueError(
            f"No script template for method '{method_id}' — supported templates are "
            f"{sorted(p.stem.replace('.py', '') for p in _TEMPLATES_DIR.glob('*.j2'))}"
        )
    tpl = _env().get_template(f"{tpl_name}.py.j2")
    return tpl.render(model_id=model_id, options=options or {})


@tool
def generate_script(method_id: str, model_id: str, options: dict | None = None) -> str:
    """Render a runnable Python quantization script for the chosen method and write it to ./out/.

    Args:
        method_id: Catalog id returned by recommend_quantization (e.g. 'awq', 'gptq', 'bnb_nf4', 'hqq').
        model_id: HuggingFace model id (e.g. 'meta-llama/Llama-3-8B').
        options: Optional dict of method-specific overrides (e.g. {'group_size': 128, 'bits': 4}).

    Returns the path of the written script. The script is meant to be executed by the user
    on a machine with a GPU and the corresponding quantization library installed.
    """
    code = render(method_id, model_id, options)
    s = load_settings()
    s.output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"quantize_{_slug(model_id)}_{method_id}.py"
    path = s.output_dir / filename
    path.write_text(code)
    return f"Wrote {path} ({len(code)} bytes). Run it with: python {path}"


__all__ = ["render", "generate_script"]
# keep resources import referenced for future package-install path resolution
_ = resources
