# Pipeline architecture

The pipeline follows a **deterministic core, agentic edges** rule. Facts that can
be evaluated mechanically are enforced by normal code; language models are used
for incomplete repository/paper evidence, script authoring, diagnosis, and human
explanations.

## Model-provider boundary

All language-model stages are constructed through `src/quant_agent/llm.py` and use
OpenAI's Responses API with response storage disabled (`store=False`). The default
policy uses `gpt-5.6-terra` for research, repository planning, script authoring,
tuning, and hyperparameter extraction. The higher-reasoning port and repair stages
use `gpt-5.6-sol`. This boundary prevents provider configuration from being spread
through agent modules and makes model/cost policy independently testable.

Global `QUANT_AGENT_MODEL` and `QUANT_AGENT_REASONING_EFFORT` overrides are
supported. Stage-specific `QUANT_AGENT_<STAGE>_MODEL` and
`QUANT_AGENT_<STAGE>_REASONING_EFFORT` take precedence.

## Research and selection

1. Resolve the model and target hardware.
2. Build a `CompatibilityRequest` from model architecture, parameters, GPU facts,
   requested bit width/backend, and calibration/QAT policy.
3. Evaluate every catalog method with `compatibility.evaluate_catalog`.
4. Give each method one of four statuses:
   - `blocked`: a hard fact failed and the Research model cannot override it;
   - `eligible`: all hard constraints pass and relevant family evidence exists;
   - `port_required`: hard constraints pass, but upstream does not document the
     target model family; the method remains selectable;
   - `unknown`: nothing failed, but non-family evidence is incomplete.
5. The Research model investigates unknowns, chooses 3–8 finalists, and explains
   tradeoffs. Schema validation and a second deterministic check reject any
   blocked finalist.

The product catalog lives in `src/quant_agent/data/methods.yaml`. Hardware/model
capability evidence and its primary-source provenance live separately in
`src/quant_agent/data/method_capabilities.yaml`.

## Staged Adapt pipeline

Adapt is split into bounded stages with typed artifacts:

1. **prepare** — choose exact script and output paths;
2. **acquire** — clone only the catalog-pinned repository and record its commit;
3. **plan** — inspect repository documentation/source and emit an `AdaptPlan`;
4. **environment** — execute the restricted install plan in the method venv;
5. **architecture** — fetch model config and inspect the meta-device module tree;
6. **port** — when required, compare upstream architecture assumptions with the
   target tree and write a content-addressed unified-diff overlay;
7. **generate** — author a script from the immutable plan and architecture facts;
8. **validate** — fail-closed staged validation;
9. **promote** — atomically replace the stable output only after validation.

The port overlay lives under `out/overlays/<method>/<model>/<patch-hash>/` with
`overlay.patch` and `manifest.json`. At launch the executor verifies the base commit,
snapshots and hashes the overlay under `jobs/<id>/overlay/`, creates a temporary
detached Git worktree, and checks/applies the patch there. Generated code receives the
patched checkout through `QUANT_AGENT_METHOD_REPO`; the canonical
`.venvs/<method>/repo` remains unchanged. Repair edits cannot remove this contract.

Every attempt writes `<script>.adapt.json`, which records completed/failed stages
without credentials. A failed attempt never promotes its temporary script.

## Validation

Generated scripts pass these stages:

1. AST parsing;
2. top-level import probing inside the method venv;
3. static semantic proof that the exact model id and output directory occur;
4. type-sensitive proof that tune-locked hyperparameters occur as named values;
5. an optional, explicit argv-only smoke command with a bounded timeout.

The smoke hook is disabled by default because a valid command depends on the
method adapter and environment. Verified adapters should configure it with a tiny
public model or a repository-provided smoke entrypoint. It never invokes a shell.

## Execution and reproducibility

`ExecutionPolicy.host()` preserves the current acknowledged host behavior.
`ExecutionPolicy.containerized(ContainerCommandPlan(...))` supports a caller-owned
Docker, Podman, nerdctl, Apptainer, or Singularity policy. The plan is argv-based,
uses a fixed placeholder allowlist, and safely quotes every rendered argument.

Every job writes `jobs/<id>/reproducibility.json` atomically. It includes the
script hash, model/method ids, output path, execution mode, platform/Python,
best-effort package and CUDA/GPU details, the cloned method commit, and any overlay
snapshot hash. The record is secret-free and referenced by `JobMeta.manifest_path`.

Container execution is an opt-in policy API rather than a security claim about an
arbitrary image. A production deployment must provide a pinned image, read-only
caches, explicit writable mounts, GPU/resource limits, and its desired network and
credential policy.
