---
name: quant-execute
description: Launch, monitor, diagnose, and repair a validated quantization artifact on a CUDA host using the canonical job registry, detached execution, overlay worktree, reproducibility manifest, and one-fix-per-attempt policy. Use for requests to execute a generated quantization script, inspect or kill a job, resume a failed job, or repair and relaunch without changing methods.
---

# Quant Execute

Read [../quant/reference/pipeline_contract.md](../quant/reference/pipeline_contract.md)
before acting. Run only on the CUDA worker. This skill consumes a validated `quant`
artifact and preserves the Python executor's storage and security contracts.

## Preconditions

- Require `nvidia-smi`, the generated stable script, `.venvs/<method>/bin/python`,
  and `.venvs/<method>/repo`.
- Determine `method_id`, exact `model_id`, output directory, hyperparameters, and
  optional overlay from the script/Adapt trace. Never infer the method solely from
  an ambiguous filename.
- Require explicit acknowledgement before running third-party/generated code on the
  host. Do not silently fall back to CPU.
- For a gated model, require HF authentication in the parent process. Never request
  or expose token values in chat and never read credential-file contents.

## Launch canonically

Never hand-create job ids, `meta.json`, manifests, overlay worktrees, or shell
wrappers. Launch through the bundled helper:

```bash
python .agents/skills/quant-execute/scripts/launch_job.py <script> \
  --method-id <method> --model-id <model> --output-dir <output> \
  --hyperparameters-json '<json>' --allow-unsafe-host-execution
```

The helper calls `executor.launch`, which:

- snapshots `script.py` under a validated timestamp/random job id;
- validates and snapshots any overlay, records its directory hash, creates a
  detached worktree at the manifest base commit, and applies the patch there;
- writes `reproducibility.json` before launch;
- starts the process in a new session with credential allowlisting;
- records pid/pgid and complete `JobMeta` in `meta.json`;
- writes stdout/stderr and an exit-code sentinel.

Report the job id and monitor commands immediately:

```bash
quant-agent jobs status <job_id>
quant-agent jobs logs <job_id> -n 200
```

## Monitor

Use `quant-agent jobs status` and `jobs logs`; do not implement a second status
machine. Poll long GPU runs around every 30 seconds and give concise progress.

- `completed`: confirm output exists, report size, manifest path, metrics if any,
  overlay snapshot/hash if any, and the complete parent/repair chain.
- `failed`: enter the repair policy below.
- `killed` or `timeout`: report the terminal reason and stop unless the user asks for
  a deliberate relaunch.
- Use `quant-agent jobs kill <job_id>` for cancellation so process-group identity and
  overlay worktree cleanup remain safe.

## Repair a failed job

Stay on the selected method. Read `meta.json`, the last 200 log lines, the saved
script, repository evidence, and every ancestor's `fix_note` plus error signature.
If the latest signature matches its parent, the prior fix did not work.

Apply exactly one new fix per attempt, with at most the requested repair budget
(the pipeline default is three):

- missing package or version conflict: change only the method venv;
- missing repository CUDA extension: build the specific documented extension with
  the detected compute capability;
- wrong script API/flag: edit only the job script after verifying repository source;
- gated-model auth, OOM at the same configuration, incompatible hardware, disk full,
  or an unclassified unsafe failure: non-retryable until external state changes.

Do not alter TUNE-LOCKED values in a repair. Do not switch methods. Do not combine
venv surgery and a script edit in one attempt.

After one fix, validate the edited script with the `quant` validator and relaunch:

```bash
python .agents/skills/quant-execute/scripts/launch_job.py jobs/<failed>/script.py \
  --method-id <method> --model-id <model> --output-dir <output> \
  --parent-job-id <failed> --attempt <n> --fix-note '<specific change>' \
  --hyperparameters-json '<unchanged-json>' \
  [--overlay-source jobs/<failed>/overlay] --allow-unsafe-host-execution
```

For ported jobs, keep the overlay header unchanged and pass the prior job's validated
`meta.overlay_path` as `--overlay-source`. The launcher snapshots it again and rebuilds
a fresh detached worktree; never patch the canonical checkout.

## Success handoff

Report the surviving output directory, disk size, final job status, complete repair
chain, `reproducibility.json`, method commit, and overlay hash. Successful runs remain
under `jobs/<id>/`; do not upload, delete, or prune them here.

If the original request included tuning, hand the completed job id to `quant-tune`.
Otherwise stop.

## Bundled resource

- `scripts/launch_job.py` — canonical executor launch wrapper.
