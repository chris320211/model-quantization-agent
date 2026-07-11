from __future__ import annotations

import json

from quant_agent import tune_history


def test_query_prior_wins_returns_only_non_dominated_entries(tmp_path, monkeypatch):
    history = tmp_path / "history.jsonl"
    monkeypatch.setattr(tune_history, "_HISTORY_PATH", history)
    rows = [
        {
            "model_id": "m", "instance_type": "g", "method_id": "awq",
            "hyperparameters": {"x": 1},
            "metrics": {"prefill_ms": 20, "decode_ms": 20, "vram_gb": 20, "ppl": 20},
            "timestamp": "2026-01-01T00:00:00+00:00", "note": None,
        },
        {
            "model_id": "m", "instance_type": "g", "method_id": "awq",
            "hyperparameters": {"x": 2},
            "metrics": {"prefill_ms": 10, "decode_ms": 10, "vram_gb": 10, "ppl": 10},
            "timestamp": "2026-01-02T00:00:00+00:00", "note": None,
        },
    ]
    history.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    wins = tune_history.query_prior_wins(
        model_id="m", instance_type="g", method_id="awq"
    )
    assert [entry.hyperparameters for entry in wins] == [{"x": 2}]
