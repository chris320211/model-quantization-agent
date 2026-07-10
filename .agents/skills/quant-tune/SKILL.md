---
name: quant-tune
description: Close the loop after a successful quantization run — measure latency / VRAM / perplexity, then iterate hyperparameters until Pareto-best or stagnation. Use when the user asks to tune, optimize, or sweep a quantized model (phrasings like "tune the awq result", "now find a better config", "optimize the gptq job", "run the tune loop on `<job_id>`"). Requires a CUDA GPU on the host (intended to run on the EC2 box, not a laptop).
---

# Quant-Tune — closed-loop hyperparameter tuner

This skill is the third leg of the `model-quantization-agent` pipeline. Its input is a completed quantization job under `jobs/<id>/` (from `quant-execute`). Its output is a Pareto-best hyperparameter configuration measured against the fp16 reference and the iteration history, plus an updated `jobs/<best_id>/metrics.json` and an entry in `~/.cache/quant-agent/tune_history.jsonl`.

It mirrors the Python package's `tune_agent.py` (single-shot proposal call), `measurement.py` (latency + VRAM + ppl probe), `pareto.py` (epsilon-tolerant dominance + stagnation), and `tune_history.py` (cross-run JSONL).

## Preconditions

- `nvidia-smi` resolves on the host. Bail if not.
- A successful job exists at `jobs/<job_id>/` with `meta.json` `status == "completed"` and an on-disk model at `meta.output_dir`.
- The method venv at `.venvs/<method_id>/bin/python` works (built by `quant-execute`).

## When to invoke

Trigger on any of:

- "tune the `<method>` result"
- "tune job `<job_id>`"
- "now find a better config" / "optimize the result"
- "run the tune loop on `<job_id>`"
- Auto: invoke immediately after `quant-execute` reports success **only if the user originally asked to tune**. Otherwise wait for an explicit request — tuning eats minutes per iteration and budget per quantize call.

If the user gives no `job_id` and `jobs/` has multiple `status: "completed"` entries, ask which one. Don't pick silently.

## Inputs

- **`job_id`** — the baseline job whose hyperparameters we'll iterate on. Must be `status: "completed"`.
- **Optional `--max-iter`** — hard cap on tune iterations including the baseline (default 5).
- **Optional `--stagnate-after`** — stop after N consecutive non-improvements (default 2).

## Phase 0 — Resolve baseline

1. `Read jobs/<job_id>/meta.json`. Confirm `status == "completed"`. Pull `method_id`, `model_id`, `output_dir`, `hyperparameters` (may be `null` for the very first baseline).
2. Resolve the method's tunable ranges:
   - First check `../quant/reference/methods.yaml` for the method's `hyperparameters_default` block. Each entry there has the form `name: {values: [...], default: ...}` — use it verbatim.
   - If the catalog has no `hyperparameters_default` for this method (rare — only some research repos), `WebFetch` the method's README and extract a flat list of ≤6 tunable knobs with explicit `values` lists (3-5 values per knob, no min/max ranges).
   - If the README extraction yields nothing usable, tell the user the method has no tunable knobs and stop after running the baseline measurement (Phase 1).
3. If `meta.hyperparameters` is null, treat the baseline config as `{name: default}` for every knob in the resolved ranges.

## Phase 1 — Baseline + fp16 measurement

Two measurements run **once** before the loop. Both write `metrics.json` next to the job dir.

### 1a — Quantized baseline

If `jobs/<job_id>/metrics.json` already exists with the four fields below, skip this. Otherwise (note the env loader — even though the quantized baseline reads weights from local disk, `transformers` may still touch HF for `chat_template.jinja` or revision metadata; sourcing is cheap insurance):

```bash
source /home/ubuntu/model-quantization-agent/.Codex/skills/_shared/load_env.sh || exit 1
cp .Codex/skills/quant-tune/reference/measure.py jobs/<job_id>/measure.py
chmod +x jobs/<job_id>/measure.py
MEASURE_MODEL_PATH=<output_dir> \
MEASURE_OUTPUT_JSON=jobs/<job_id>/metrics.json \
.venvs/<method_id>/bin/python jobs/<job_id>/measure.py \
  > jobs/<job_id>/measure.log 2>&1
```

Read `metrics.json` — it must contain `prefill_ms`, `decode_ms`, `vram_gb`, `ppl`. If the run failed (exit non-zero, or no JSON), tail `measure.log` and surface the error. Common failures:

- `ModuleNotFoundError: datasets` → `Bash: .venvs/<method_id>/bin/pip install datasets` and retry.
- `CUDA out of memory` during ppl → reduce `MEASURE_MAX_LENGTH` from 2048 to 1024 (env var). Don't reduce stride.
- `RuntimeError: CUDA GPU required for measurement` → bail, this skill only runs on GPU.

Update `meta.json`: set `metrics` to the four fields, `hyperparameters` to the baseline config from Phase 0. Persist.

### 1b — fp16 reference (cached)

The fp16 baseline is the apples-to-apples reference for "did we beat the unquantized model". It's expensive (loads the model in fp16) so cache by `(model_id, instance_type)`.

1. Read `~/.cache/quant-agent/fp16_baselines.json` (create empty `{}` if missing).
2. Key: `<model_id>::<instance_type>` (use `unknown` for instance if you can't tell — extract from `nvidia-smi --query-gpu=name --format=csv,noheader` then map via `../quant/reference/aws_instances.yaml`).
3. Hit → use cached metrics, skip 1b. Miss → continue.

Build the fp16 reference venv if missing (one time):

```bash
if [ ! -f .venvs/_fp16_reference/.installed ]; then
  python3.10 -m venv .venvs/_fp16_reference
  source .venvs/_fp16_reference/bin/activate
  pip install --upgrade pip wheel
  pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
  pip install 'transformers<5.0' accelerate safetensors sentencepiece datasets  # transformers 5.x requires torch>=2.4
  deactivate
  touch .venvs/_fp16_reference/.installed
fi
```

(Use `cu124` and `torch==2.4.1` if `nvidia-smi --query-gpu=compute_cap` ≥ 9.0 — Hopper.)

Then run the measure script with the model id (HF will load fp16 directly — `HF_TOKEN` is required for gated repos, hence the loader):

```bash
source /home/ubuntu/model-quantization-agent/.Codex/skills/_shared/load_env.sh || exit 1
mkdir -p jobs/<job_id>/fp16_baseline
cp .Codex/skills/quant-tune/reference/measure.py jobs/<job_id>/fp16_baseline/measure.py
MEASURE_MODEL_PATH=<model_id> \
MEASURE_OUTPUT_JSON=jobs/<job_id>/fp16_baseline/metrics.json \
.venvs/_fp16_reference/bin/python jobs/<job_id>/fp16_baseline/measure.py \
  > jobs/<job_id>/fp16_baseline/measure.log 2>&1
```

Read the metrics. Append to `~/.cache/quant-agent/fp16_baselines.json`:

```json
{ "<model_id>::<instance_type>": {"prefill_ms": ..., "decode_ms": ..., "vram_gb": ..., "ppl": ...} }
```

Best-effort — if 1b fails for any reason (network, OOM on fp16), proceed without an fp16 reference. The iteration-vs-iteration Pareto check still works.

## Phase 2 — Tune loop

Hold three lists in your context as you iterate:

- `history`: list of `{iter, hyperparameters, metrics, note}` records, starting with the baseline.
- `iter_jobs`: list of `job_id`s, parallel to `history`. Used for disk pruning.
- `best_so_far`: the running Pareto-best metrics (initialize to baseline).

For iteration `N` from `2` to `--max-iter`:

### 2a — Propose the next config

You play `tune_agent.propose`. Build the proposal yourself by reading the inputs and emitting one of:

```yaml
decision: propose
hyperparameters: {name: value, ...}
rationale: <one-sentence reason>
```

or

```yaml
decision: stop
reason: <one-sentence reason>
```

**Inputs you have:**

- The method's `ranges` (every knob's allowed `values` list).
- `history` so far (every prior config + its metrics, or `null` if the iteration crashed).
- `best_so_far`.
- `fp16_baseline` (or `null` if Phase 1b skipped).
- `prior_wins`: `tail -200 ~/.cache/quant-agent/tune_history.jsonl | jq 'select(.model_id=="<id>" and .instance_type=="<type>" and .method_id=="<id>")' | head -5` — warm-start hints from past sessions.

**Decision rules** (lifted from `src/quant_agent/tune_agent.py:_PROMPT`):

- `propose`: every key must be a name from `ranges`, every value must be from that knob's `values` list. Do **not** propose a configuration already in `history` (exact dict match — repeats waste budget).
- `stop`: choose this when (a) every reasonable Pareto-improving move has been tried, (b) the search space is exhausted, (c) the last 2 iterations regressed clearly, or (d) the running best already strictly dominates fp16 on speed AND VRAM with negligible ppl drift.
- Optimize for Pareto improvement: **any metric better by >2% AND no metric worse by >2%** (stricter for ppl: >0.5%). Avoid fads — small changes to one knob beat large changes to several at once.

**Constraint check (you must do this yourself before applying):**

- Every key is in `ranges` → if not, switch to `stop` with reason `proposed unknown knob <name>`.
- Every value is in that knob's `values` → if not, switch to `stop` with reason `proposed <name>=<v> not in allowed values <list>`.
- The proposal isn't a duplicate of any prior `history` config → if dup, switch to `stop` with reason `proposal duplicates a prior iteration`.

If the decision is `stop`, exit the loop and go to Phase 4. Otherwise continue.

### 2b — Re-emit the script with the new config

You re-write the quantize script for this iteration. The convention is `out/quantize_<safe_model>_<method_id>_iter<N>.py`, derived from the baseline script's name. Copy the baseline script and:

1. Replace the `# TUNE-LOCKED HYPERPARAMETERS` block at the top with the new config (one `# name=value` line per key). If the baseline script lacks that block, prepend one:

   ```python
   # TUNE-LOCKED HYPERPARAMETERS (do not modify in fix loop):
   # group_size=64
   # zero_point=True
   ```

2. Find every site in the script that consumes a tuned knob (e.g. `quant_config = BaseQuantizeConfig(group_size=128, zero_point=True)` for AWQ, or `--group-size 128` in a Style B subprocess) and update them to the new values.

3. Validate: `python3 -c "import ast; ast.parse(open('<path>').read())"`. If parse fails, fix once with `Edit` and re-validate.

This step is fragile because the baseline script's structure varies by method. If you can't locate the call site for a tuned knob (e.g. the script hard-codes a value the LLM didn't produce), fall back to **regenerating** the script from scratch using the same logic as the `quant` skill's Phase 4 — but with `chosen.hyperparameters` set to the new config. Pull `chosen` from `jobs/<job_id>/meta.json`.

### 2c — Launch and supervise

Run **Phase 3 of `quant-execute`** to launch the new script as a background job. Set:

- new `job_id` (timestamp + random hex)
- `parent_job_id` = `<baseline job_id>`
- `attempt = 1`
- `tune_iter = N`
- `hyperparameters = <new config>`

Then run **Phase 4 of `quant-execute`** to wait for completion. If it fails, run **Phase 5 (fix loop) of `quant-execute`** with `max_attempts = 3`, exactly as if it were a fresh user-launched job.

If the iteration ends `failed` after fix attempts: append `{iter: N, hyperparameters: <config>, metrics: null, note: "crashed after fix loop"}` to `history`, append the final `job_id` to `iter_jobs`, prune via Phase 3, and continue to the next iteration.

If the iteration ends `killed`: same handling — record as null metrics, note `"killed"`.

### 2d — Measure

Re-run Phase 1a (the quantized measurement) on the new completed job. Read `metrics.json`. If it failed, record null metrics and continue.

Append to `history`:

```json
{ "iter": <N>, "hyperparameters": <config>, "metrics": {...}, "note": null }
```

### 2e — Update best, persist cross-run history, prune

1. **Pareto check.** Apply the dominance rule to `(best_so_far, current_metrics)`. The rule (from `src/quant_agent/pareto.py`):

   ```
   epsilons = {prefill_ms: 0.02, decode_ms: 0.02, vram_gb: 0.01, ppl: 0.005}
   delta(name) = (curr[name] - prev[name]) / abs(prev[name])
   any_better = any(delta < -eps for name)
   any_worse  = any(delta >  eps for name)
   improvement = any_better and not any_worse
   ```

   If `improvement`, set `best_so_far = current_metrics`, remember the iteration index.

2. **Cross-run persistence.** Append to `~/.cache/quant-agent/tune_history.jsonl` (one JSON per line, no array wrapping). Strip absolute paths (replace `$HOME` with `~`):

   ```json
   {"model_id": "...", "instance_type": "...", "method_id": "...", "hyperparameters": {...}, "metrics": {...}, "timestamp": "<ISO-8601 UTC>", "note": "iter<N>"}
   ```

3. **Disk prune.** Keep only the latest iteration and the running best. Delete the rest:
   - `keep = {iter_jobs[-1]} ∪ {iter_jobs[idx_of_best]}`
   - For every other `iter_jobs[i]`: `rm -rf jobs/<id>/` and `rm -rf <output_dir>` (only if it lives under `./quantized/`).

   **Order matters.** Prune **after** appending to `tune_history.jsonl` so the data survives even if `rm` races with anything else.

### 2f — Stagnation check

Apply `detect_stagnation(history, n=stagnate_after)`:

- If `len(history) <= n`: not stagnant.
- Otherwise: take the prefix `history[:-n]`, compute `best_so_far` over that prefix. Then check whether **any** of the last `n` iterations Pareto-improves over that prefix-best. If none do, you're stagnant — exit the loop.

## Phase 3 — Final report

Print a summary like:

```
=== Tune summary: AWQ (awq) ===
fp16 baseline:  prefill=812.3ms  decode=4123.4ms  vram=14.20GB  ppl=6.812
  iter 1: prefill=412.5ms  decode=2087.1ms  vram=4.71GB  ppl=6.901   hp={group_size: 128, zero_point: true}
* iter 2: prefill=389.4ms  decode=1953.0ms  vram=4.69GB  ppl=6.890   hp={group_size: 64, zero_point: true}
  iter 3: crashed   hp={group_size: 32, zero_point: false}
  iter 4: prefill=391.0ms  decode=1968.2ms  vram=4.70GB  ppl=6.895   hp={group_size: 64, zero_point: false}
Best config: iter 2  {group_size: 64, zero_point: true}
Result: Pareto-improves over fp16 baseline.
```

Mark the best iteration with `*`. Print the path to the surviving `jobs/<best_id>/` and `<output_dir>` so the user can use the model.

## Guardrails

- **One method only.** Don't switch methods between iterations — the user already picked. If the chosen method appears genuinely untunable (every config crashes), say so and stop. Don't fall back to a different method.
- **Don't propose values outside `ranges.values`.** Even if you think a value would help, the catalog enumeration is the contract. Out-of-range values become a hard `stop`, not a clever cast.
- **Don't propose a configuration already in `history`.** Each iteration must explore a new point. If you can't find an unexplored point, `stop`.
- **Don't prune before persisting.** Always append to `tune_history.jsonl` first, then `rm -rf`. Pruning the in-flight job's artifacts is a hard "no".
- **Don't measure faster than the script can produce metrics.** WikiText-2 perplexity takes minutes; don't poll the measurement subprocess at sub-30s intervals.
- **Don't auto-retry non-retryable measurement failures.** A `RuntimeError: CUDA GPU required` or `disk full` is a bail, not a fix-loop entry.
- **Don't tune the fp16 reference.** It's the unquantized ground truth and is intentionally not in `ranges`.

## Cross-skill contract with `quant`, `quant-setup`, and `quant-execute`

Inputs this skill expects:

- `jobs/<job_id>/` directory with `meta.json` `status == "completed"` and a populated `output_dir`.
- `.venvs/<method_id>/bin/python` from `quant-execute`'s Phase 2.
- `out/quantize_<safe_model>_<method_id>.py` from `quant`'s Phase 4 (used as the iteration template).
- `.env` (mode 600) populated by `quant-setup`. Required for `HF_TOKEN` whenever 1b (fp16 reference) needs to load a gated model. Both Phase 1a and Phase 1b source `_shared/load_env.sh`. If the loader returns non-zero, bail and direct the user to `/quant-setup` rather than retry.

Outputs this skill produces:

- `jobs/<best_job_id>/metrics.json` with the four-field schema.
- `~/.cache/quant-agent/tune_history.jsonl` (append-only).
- `~/.cache/quant-agent/fp16_baselines.json` (when 1b succeeded).
- `out/quantize_<safe_model>_<method_id>_iter<N>.py` files for each iteration that wrote a script.
- A printed summary table of the Pareto front.

## Reference files (bundled in this skill)

- `reference/measure.py` — the standalone measurement script. Reads `MEASURE_MODEL_PATH` and `MEASURE_OUTPUT_JSON` from env, runs prefill (2k+128) and decode (32+512) latency probes, captures `torch.cuda.max_memory_allocated`, and computes WikiText-2 sliding-window perplexity (max_length=2048, stride=512). Identical to the embedded `MEASURE_SCRIPT` in `src/quant_agent/measurement.py`. Drop into the job dir, set the two env vars, run it.
