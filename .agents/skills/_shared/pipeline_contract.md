# Pipeline contract (skills-first)

User inputs: **method name**, **model name**, **GPU instance**. No method catalog.

Canonical helpers live in `.agents/skills/_shared/scripts/`. Every stage skill
invokes those scripts; do not reimplement clone, overlay, launch, or verify.

Order:

1. `quant-setup` (parent only, if HF auth is missing)
2. `quant-gather` subagent — paper + GitHub + clone + HF snapshot + GPU facts + **one** venv install
3. `quant-port` coordinator — up to three named strategy subagents; validate; pick a winner (no GPU yet)
4. `quant-run` subagent — launch with retry loops
5. `quant-verify` subagent — saved artifact must generate (not Hub fp16)
6. Baseline compare on this GPU
7. `quant-kernel` subagent — only if quantized did not help, or the user asked
8. Tell the user where weights and metrics live

## Request JSON

Handoff file: `out/requests/<slug>.json` (no secrets). Later stages read only that
file plus job ids it names.

Required keys:

| key | meaning |
| --- | --- |
| `method_name` | User method string |
| `model_id` | Hugging Face `org/model` |
| `gpu_instance` | e.g. `g5.xlarge` |
| `arxiv_id` | arXiv id from gather |
| `paper_path` | `.cache/papers/<id>.txt` |
| `repo_url` | `https://github.com/owner/repo` |
| `repo_path` | `.venvs/<slug>/repo` |
| `repo_commit` | 40-char HEAD SHA |
| `hf_snapshot_path` | `.cache/hf-snapshots/...` |
| `slug` | Path-safe id used for venv, overlays, jobs |

Optional keys gather may add: `instance_type`, `gpu_arch`, `vram_gb`, `gpu`,
`venv_python`. Port may add `winner_overlay_dir`, `winner_script`, `ranked`.

## Layout

- Canonical clone: `.venvs/<slug>/repo` (never edited)
- Venv python: `.venvs/<slug>/bin/python`
- Overlays: `out/overlays/<slug>/<strategy>/<hash>/`
- Jobs: `jobs/<job_id>/`
- Weights: `./quantized/<slug>`

## Invariants

- Never edit the canonical method clone; overlays only, applied to a temp worktree.
- `quant-setup` never runs inside a subagent. Never paste tokens into prompts.
- Installers, git, dry-import, and measurements receive no cloud credentials.
- Quantization and verify receive HF auth only when the model is gated.
- Host-mutating scripts require `--allow-unsafe-host-execution`.
- One GPU job at a time. Port fans out **validate-only** workers.
- Verify must reload the **saved** quantized artifact, not the Hub snapshot.
  `--baseline` smokes the original snapshot as fp16 and records `improved_vs_fp16`.
