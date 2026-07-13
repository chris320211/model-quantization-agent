---
name: quant
description: Research quantization methods and produce a validated HuggingFace quantization artifact for a target model and AWS/NVIDIA GPU. Use for requests to port or quantize a model, compare methods, select a method, generate a script, or create a separate model-family overlay. Runs the current deterministic compatibility and staged Adapt contracts without directly calling the Python OpenAI backend; stops before launching the GPU job.
---

# Quant

Use the active Codex session for reasoning and the Python package for deterministic
contracts. Read [reference/pipeline_contract.md](reference/pipeline_contract.md)
before acting. This skill ends with a validated script and optional overlay; hand
execution to `quant-execute` and optimization to `quant-tune`.

Do not call `quant-agent ask` when the user requested subscription-backed operation:
that command uses the Python OpenAI API backend. Do not read or edit credential files.

## Resolve the request

Collect the exact model and target hardware plus any requested bit width, backend,
quality/speed priority, calibration availability, QAT permission, activation
quantization, KV-cache quantization, and `trust_remote_code` consent.

- Resolve exact `org/model` ids directly. Otherwise use
  `reference/model_aliases.yaml`; fuzzy HF search results require user confirmation.
- Resolve AWS facts from `aws_instances.yaml` plus `gpu_specs.yaml`. On the target
  machine, merge live `nvidia-smi` free/total VRAM and driver data.
- Fetch HF `config.json` and model metadata to obtain architectures and parameter
  count. Report whether the count is exact, file-size-derived, or approximate.
- Ask for a missing model id or hardware target when it materially changes fitting.

## Research every method

Read `reference/methods.yaml` and `reference/method_capabilities.yaml`. Run the
canonical evaluator, passing only known flags:

```bash
python .agents/skills/quant/scripts/evaluate_compatibility.py \
  --params-b <params_b> --vram-gb <vram_gb> \
  --compute-capability <cc> --gpu-arch <arch> \
  --architecture <ArchitectureClass> [request flags]
```

Use `--calibration unavailable`, `--allow-qat`, `--need-activation-quant`,
`--need-kv-cache-quant`, `--target-bits`, and `--backend` only when requested or
known. Never reproduce old blanket sub-4-bit rules; capability data and the
deterministic evaluator are authoritative.

Produce exactly one considered row for every catalog id:

- `blocked`: reject with the deterministic reason; never promote it.
- `eligible`: hard constraints passed; investigate only for ranking evidence.
- `port_required`: may be included; state that Adapt must create a reviewable
  overlay because native family support is not established.
- `unknown`: investigate repository/paper evidence conservatively.

Return 3–8 distinct finalists copied from the catalog. Use the evaluator's chosen
bits, `params_b * bits / 8 * 1.4` VRAM estimate, catalog scores/repository, and
catalog hyperparameter defaults. Explain tradeoffs without selecting a winner.
Ask the user to select a finalist unless they already named one explicitly.

## Run staged Adapt on the selected method

Keep the selected method locked throughout Adapt. Record each stage in
`out/quantize_<safe_model>_<method>.adapt.json`; persist a failed-stage record when
anything fails.

1. **Prepare** — choose the stable script path, temporary attempt path, and exact
   output directory from `quant_agent.executor.default_output_dir`.
2. **Acquire** — clone only the catalog repository into
   `.venvs/<method>/repo` and record HEAD:

   ```bash
   python .agents/skills/quant/scripts/method_env.py \
     --allow-unsafe-host-execution clone \
     --method-id <method> --repo-url <catalog-url>
   ```

3. **Plan** — inspect README plus relevant source and emit one `AdaptPlan`:
   repository-supported install steps, `standalone` or `wrapper`, optional relative
   entrypoint, `native|port_required|unknown` support, evidence files, and support
   evidence. Repository code decides import names and flags; never invent them.
4. **Environment** — after explicit host-execution acknowledgement, build the
   canonical method venv with the plan's restricted Python/pip steps:

   ```bash
   python .agents/skills/quant/scripts/method_env.py \
     --allow-unsafe-host-execution install --method-id <method> \
     --step '<repository-supported pip/python step>'
   ```

   Repeat `--step` as needed. The helper chooses the local torch/CUDA spec and
   strips cloud credentials from installers.
5. **Architecture** — fetch full config and use the method venv to instantiate the
   model on the meta device, then collapse `named_modules()` into target patterns.
   If custom code is required and not approved, remain config-only and say so.
6. **Port** — if Research or repository evidence says porting is required, compare
   upstream dispatch/target assumptions with the architecture evidence. Create the
   smallest text-only unified diff without editing the canonical checkout:

   ```bash
   python .agents/skills/quant/scripts/write_overlay.py \
     --method-id <method> --model-id <model> --base-commit <40-char-head> \
     --patch-file <temporary-diff> --rationale '<why>' \
     --evidence-file <repo-file> --target-module <module-pattern>
   ```

7. **Generate** — author one script from the immutable plan, exact model/output,
   architecture evidence, and locked hyperparameters. A wrapper must use the cloned
   repository entrypoint and argv. A ported script must carry the exact overlay
   header and read both executor-provided overlay environment variables.
8. **Validate** — run the canonical staged validator, up to three authoring attempts:

   ```bash
   python .agents/skills/quant/scripts/validate_script.py <attempt-script> \
     --method-id <method> --model-id <model> --output-dir <exact-output> \
     --hyperparameters-json '<json>' [--overlay-dir <bundle>] \
     --allow-unsafe-host-execution
   ```

9. **Promote** — atomically replace the stable script only after validation passes.
   A failed attempt never becomes the handoff artifact.

## Handoff

Report:

- selected method, bit width, model, target, script path, and output directory;
- Adapt trace path and recorded repository commit;
- overlay bundle/hash when present;
- validation checks and any skipped smoke stage;
- that `quant-execute` is the next skill for launch/repair.

Do not run the generated script in this skill. Do not silently fall back to another
method after the user selects one. Do not mutate the canonical method checkout.

## Bundled resources

- `reference/methods.yaml` — canonical product catalog mirror.
- `reference/method_capabilities.yaml` — compatibility facts and provenance mirror.
- `reference/model_aliases.yaml`, `aws_instances.yaml`, `gpu_specs.yaml` — resolution.
- `scripts/evaluate_compatibility.py` — canonical deterministic gate.
- `scripts/method_env.py` — catalog-locked clone and dynamic method venv build.
- `scripts/write_overlay.py` — validated content-addressed overlay writer.
- `scripts/validate_script.py` — canonical staged validation and overlay contract.
