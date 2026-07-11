# Pipeline architecture

The pipeline follows a **deterministic core, agentic edges** rule. Facts that can
be evaluated mechanically are enforced by normal code; language models are used
for incomplete repository/paper evidence, script authoring, diagnosis, and human
explanations.

## Research and selection

1. Resolve the model and target hardware.
2. Build a `CompatibilityRequest` from model architecture, parameters, GPU facts,
   requested bit width/backend, and calibration/QAT policy.
3. Evaluate every catalog method with `compatibility.evaluate_catalog`.
4. Give each method one of three statuses:
   - `blocked`: a hard fact failed and the Research model cannot override it;
   - `eligible`: all hard constraints pass and relevant family evidence exists;
   - `unknown`: nothing failed, but the evidence is incomplete.
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
6. **generate** — author a script from the immutable plan and architecture facts;
7. **validate** — fail-closed staged validation;
8. **promote** — atomically replace the stable output only after validation.

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
best-effort package and CUDA/GPU details, and the cloned method commit. The record
is secret-free and referenced by `JobMeta.manifest_path`.

Container execution is an opt-in policy API rather than a security claim about an
arbitrary image. A production deployment must provide a pinned image, read-only
caches, explicit writable mounts, GPU/resource limits, and its desired network and
credential policy.
