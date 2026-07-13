# Canonical pipeline contract

Read this reference before running any quant skill. The Python package is the
source of truth; skill workflows must preserve these contracts instead of
reimplementing weaker alternatives.

## End-to-end flow

1. Research resolves the model, target hardware, and user constraints.
2. `compatibility.evaluate_catalog` assigns every catalog method one status:
   `blocked`, `eligible`, `port_required`, or `unknown`.
3. Research emits exactly one verdict per catalog id and 3–8 selectable finalists.
   A blocked method can never be a finalist. `port_required` remains selectable.
4. The user selects the method. Research never silently picks a winner.
5. Adapt runs bounded stages: `prepare`, `acquire`, `plan`, `environment`,
   `architecture`, optional `port`, `generate`, `validate`, and `promote`.
6. Execution snapshots the validated script, optional overlay, metadata, logs,
   and a secret-free reproducibility manifest under `jobs/<job_id>/`.
7. Runtime failures stay on the selected method and receive at most one distinct
   repair per attempt. Gated-model auth, OOM, disk-full, and incompatible hardware
   are non-retryable without an external change.
8. Optional tuning measures the baseline, optionally measures fp16, proposes only
   enumerated hyperparameters, re-runs Adapt, executes, measures, and retains the
   epsilon-aware Pareto frontier until stop, stagnation, or the iteration cap.

## Research facts

- `reference/methods.yaml` is authoritative for ids, repositories, bit widths,
  quantized tensors, backends, QAT/calibration flags, scores, and curated defaults.
- `reference/method_capabilities.yaml` is authoritative for hardware/model-family
  evidence and provenance. Empty lists mean unknown, not universal support.
- `reference/model_aliases.yaml`, `aws_instances.yaml`, and `gpu_specs.yaml` are
  packaged resolution data.
- Merge static hardware facts with live `nvidia-smi` data when running on the target.
- Use the canonical compatibility helper in `scripts/evaluate_compatibility.py`.
  Deterministic `blocked` results are non-overridable.
- Missing family evidence yields `port_required`, not automatic rejection, when
  hardware and algorithm constraints pass.
- Weight memory is estimated as `params_b * bits / 8 * 1.4` when both inputs exist.

## Adapt artifacts

- The catalog-pinned canonical checkout is `.venvs/<method_id>/repo`.
- The checkout origin must match a catalog repository. Record the exact HEAD commit.
- Build `.venvs/<method_id>` through the canonical repo helper. Torch/CUDA selection
  is detected by `quant_agent.tools.torch_spec`; do not hard-code a global torch pin.
- Finalize one repository-derived `AdaptPlan` with `install_steps`, `script_style`,
  optional relative `entrypoint`, `model_support`, `support_evidence`,
  `evidence_files`, and notes.
- Fetch full model config and inspect the meta-device module tree after the venv is
  built. Never enable `trust_remote_code` without explicit user approval.
- Use the paper only for the chosen method. Repository source is authoritative for
  executable APIs and flags.
- Write attempt scripts under `out/` and promote atomically only after validation.
- Persist `<script>.adapt.json` even on failure. A failed attempt never replaces the
  stable script.

## Separate port overlay

- Never edit `.venvs/<method_id>/repo` to port a model family.
- Emit a text-only unified diff through `scripts/write_overlay.py` under
  `out/overlays/<method>/<model>/<content-id>/`.
- The bundle contains `overlay.patch` and `manifest.json`, including method/model,
  base commit, patch hash, rationale, evidence files, and target modules.
- Renames, binary patches, path traversal, `.git`, and `.env*` paths are rejected.
- A ported script must include the exact `# QUANT_AGENT_OVERLAY_DIR=...` header and
  consume both `QUANT_AGENT_OVERLAY_DIR` and `QUANT_AGENT_METHOD_REPO`.
- At launch the executor snapshots and hashes the overlay, creates a detached Git
  worktree at the recorded base commit, checks/applies the patch there, and leaves
  the canonical checkout unchanged.

## Validation

Run `scripts/validate_script.py`; do not downgrade validation to AST-only.

1. Parse Python AST.
2. Probe top-level imports in the method venv with cloud credentials removed.
3. Prove the exact model id, output directory, and type-sensitive locked
   hyperparameters occur in the program.
4. Validate the overlay contract when present.
5. Run a smoke command only when an explicit safe argv contract exists.

Validation has at most three authoring attempts and fails closed. Only a validated
attempt is atomically promoted.

## Job storage and execution

Launch through `quant-execute/scripts/launch_job.py`; do not manually reproduce
`executor.launch` with ad-hoc shell metadata.

Each job can contain:

- `meta.json`: status, pid/pgid, attempt chain, tune iteration, hyperparameters,
  metrics, execution mode, manifest and overlay references;
- `script.py`, `stdout.log`, `stderr.log`, and `exit_code`;
- `reproducibility.json`: script hash, runtime versions, GPU/CUDA facts, method
  commit, execution argv, and overlay snapshot hash;
- `overlay/` and temporary `method-repo/` for ported runs;
- `metrics.json`, `measure.py`, `measure.log`, and optional `fp16_baseline/`.

Host execution requires explicit acknowledgement. Generated and third-party code
receives an allowlisted environment; OpenAI, GitHub, and unrelated cloud credentials
must not reach installers, import probes, measurements, or quantization scripts.
Only HF authentication is forwarded when model access requires it. Jobs launch in a
new session and survive SSH disconnects.

Use `quant-agent jobs list|status|logs|kill` for inspection and control.

## Tuning

- Generic HF measurement currently supports: `awq`, `gptq`, `bnb_nf4`,
  `bnb_llm_int8`, `hqq`, `autoround`, `fp8`, and `smoothquant`.
- Metrics minimize prefill latency, per-token decode latency, peak VRAM, and
  WikiText-2 perplexity. Benchmark version and measurement overrides participate
  in the fp16 cache key.
- Catalog hyperparameter enumerations win. Otherwise infer at most six meaningful
  knobs from the chosen repository and cache by method plus repository commit.
- Materialize complete configurations, enforce types and allowed values, and never
  repeat a configuration.
- Epsilons are 2% prefill, 2% decode, 1% VRAM, and 0.5% perplexity.
- Persist successful results to `~/.cache/quant-agent/tune_history.jsonl` before
  pruning. Retain the latest real iteration and every non-dominated frontier job.

## Authentication modes

- A Codex/ChatGPT-authenticated skill run uses the active Codex subscription for
  reasoning and does not require `OPENAI_API_KEY`.
- The Python `quant-agent ask` path uses the configured OpenAI API backend and does
  require `OPENAI_API_KEY`.
- HF authentication may still be required for gated models. Credentials must come
  from the process environment or the interactive setup flow, never from chat.
- Never read, print, or edit credential-file contents with agent tools.
