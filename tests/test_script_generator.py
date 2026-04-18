from __future__ import annotations

import ast
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import pytest

from quant_agent.tools.recommender import load_catalog
from quant_agent.tools.script_generator import render


@pytest.mark.parametrize(
    "method_id",
    [m["id"] for m in load_catalog() if m.get("template")],
)
def test_template_renders_and_parses(method_id):
    code = render(method_id, "meta-llama/Llama-3-8B", options={"bits": 4})
    # every generated script must be valid Python
    ast.parse(code)
    assert "meta-llama/Llama-3-8B" in code


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        render("nonexistent", "meta-llama/Llama-3-8B")


def test_method_without_template_raises():
    # smoothquant is in the catalog with template=null.
    with pytest.raises(ValueError, match="No script template"):
        render("smoothquant", "meta-llama/Llama-3-8B")


def test_awq_options_propagate():
    code = render(
        "awq", "meta-llama/Llama-3-8B", options={"group_size": 64, "bits": 4}
    )
    assert "q_group_size" in code
    assert "64" in code
