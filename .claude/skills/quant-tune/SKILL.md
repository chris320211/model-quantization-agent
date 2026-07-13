---
name: quant-tune
description: Measure and optimize a successfully quantized model on a CUDA host using the canonical benchmark, fp16 cache, bounded hyperparameter proposals, repair supervision, epsilon-aware Pareto frontier, stagnation detection, and safe pruning. Use for requests to tune, benchmark, sweep, compare configurations, or find Pareto-best settings for a completed quantization job.
---

# Quant Tune

Read [../quant/reference/pipeline_contract.md](../quant/reference/pipeline_contract.md)
before acting. Start from one completed `quant-execute` job and keep its method fixed.

## Resolve and gate tuning

Read the baseline `meta.json`, output, script, Adapt trace, repository commit, and
hyperparameters. Require CUDA and a validated measurement adapter. The current
generic-HF allowlist is `awq`, `gptq`, `bnb_nf4`, `bnb_llm_int8`, `hqq`,
`autoround`, `fp8`, and `smoothquant`; fail clearly for other formats.

Resolve tunable ranges in this order:

1. Curated `hyperparameters_default` from the method catalog.
2. Cached ranges keyed by method plus repository commit.
3. At most six meaningful knobs extracted from the chosen repository README/source,
   with explicit value lists and defaults.
4. Empty ranges: measure once and stop.

Persist resolved ranges to the baseline job as `hyperparams.yaml`. Never invent a
value outside an enumeration.

## Measure the baseline

Use the canonical measurement launcher already bundled as
`reference/measure.py`; it imports `quant_agent.measurement.MEASURE_SCRIPT`.
Prefer the package's `run_measurement` path so `metrics.json`, `measure.py`, and
`measure.log` stay in the job.

The benchmark records median prefill latency, per-token decode latency, peak allocated
VRAM, WikiText-2 perplexity, sample standard deviations, sample count, and benchmark
details. Do not substitute a one-shot timing loop.

Attempt the fp16 reference through `quant_agent.baseline.measure_fp16_baseline`.
It uses a dynamically selected torch/CUDA venv and caches by model, instance,
benchmark version, torch/CUDA spec, and measurement overrides. Treat failure as
best-effort and continue with iteration comparisons.

## Run the bounded loop

Initialize history with the completed baseline and query prior non-dominated wins for
the exact `(model_id, instance_type, method_id)` tuple.

For iterations 2 through the configured cap:

1. Build the current complete Pareto frontier.
2. Propose one configuration or stop. Every knob must be known, correctly typed, and
   in its allowed values. Materialize omitted knobs from defaults. Reject duplicates.
3. Re-run the full `quant` Adapt stages with a distinct `_iterN` script and output
   directory. Do not patch the baseline script with blind string replacement.
4. Launch through `quant-execute/scripts/launch_job.py` with `--tune-iter N` and the
   complete hyperparameter JSON.
5. Supervise the normal one-fix-per-attempt repair loop. Record failed/killed
   iterations with null metrics and continue when safe.
6. Measure a completed output through the canonical benchmark and write metrics into
   its `meta.json`.
7. Append successful metrics to `~/.cache/quant-agent/tune_history.jsonl` before any
   pruning.
8. Keep the latest real iteration plus every job on the non-dominated frontier.
   Prune only job/output paths verified under the repository's managed roots.
9. Stop on an explicit stop decision, exhausted search space, the iteration cap, or
   configured consecutive iterations that do not change the frontier.

Use the package epsilons: 2% prefill, 2% decode, 1% VRAM, and 0.5% perplexity. Lower
is better for all four metrics. An incomparable speed/quality tradeoff belongs on the
frontier; do not collapse the result to a single scalar winner.

## Final report

Print the fp16 reference when available, every iteration/config/status, all frontier
points, surviving job/output paths, and whether any frontier point Pareto-improves
over fp16. Mark crashes and measurement failures explicitly.

Do not switch methods, tune fp16, repeat a configuration, prune before persistence,
or delete a frontier job.

## Bundled resource

- `reference/measure.py` — thin launcher for the canonical benchmark implementation.
