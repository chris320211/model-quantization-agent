from __future__ import annotations

from types import SimpleNamespace

from quant_agent.tools import hf_hub


def test_safetensors_total_takes_priority():
    info = SimpleNamespace(safetensors=SimpleNamespace(total=7_000_000_000))
    params_b, source = hf_hub._resolve_params_b(info, cfg={"torch_dtype": "float16"})
    assert params_b == 7.0
    assert source == "safetensors_total"


def test_siblings_sum_fallback_uses_dtype():
    # 7B fp16 model = ~14 GB of safetensors on disk; expect ~7B params.
    info = SimpleNamespace(
        safetensors=None,
        siblings=[
            SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=7_000_000_000),
            SimpleNamespace(rfilename="model-00002-of-00002.safetensors", size=7_000_000_000),
            SimpleNamespace(rfilename="tokenizer.json", size=500_000),
        ],
    )
    params_b, source = hf_hub._resolve_params_b(info, cfg={"torch_dtype": "float16"})
    assert source == "siblings_sum"
    assert 6.9 <= params_b <= 7.1


def test_siblings_sum_fp32_halves_the_param_count():
    info = SimpleNamespace(
        safetensors=None,
        siblings=[
            SimpleNamespace(rfilename="model.safetensors", size=4_000_000_000),
        ],
    )
    params_b, source = hf_hub._resolve_params_b(info, cfg={"torch_dtype": "float32"})
    assert source == "siblings_sum"
    assert params_b == 1.0  # 4GB / 4 bytes = 1B params


def test_config_approx_when_no_siblings():
    # 32 layers, 4096 hidden → ~6.4B params via 12 * L * H^2.
    info = SimpleNamespace(safetensors=None, siblings=[])
    cfg = {"num_hidden_layers": 32, "hidden_size": 4096}
    params_b, source = hf_hub._resolve_params_b(info, cfg)
    assert source == "config_approx"
    assert 6.0 <= params_b <= 7.0


def test_all_fallbacks_fail_returns_none():
    info = SimpleNamespace(safetensors=None, siblings=[])
    params_b, source = hf_hub._resolve_params_b(info, cfg={})
    assert params_b is None
    assert source is None
