---
name: quant
description: Recommend and generate a HuggingFace model quantization script for a given AWS GPU instance. Use when the user asks to port, quantize, or run a HuggingFace model on a specific EC2 GPU type (phrasings like "port llama2 7b to g5.xlarge", "quantize Mistral 7B for an A100", "what quant method should I use for Qwen 2.5 14B on g6.xlarge"). Produces a ranked candidate list, takes the user's pick, then writes a syntax-validated Python quantization script to ./out/.
---

# Quant — model quantization research + script generator

This skill replicates the `model-quantization-agent` Python package's Research and Adapt phases without calling the Anthropic API directly. You (Claude Code) do the work the LangChain agents would have done, using the bundled catalog files in `reference/` plus your built-in `Read`, `Write`, `Edit`, `Bash`, and `WebFetch` tools.

The skill stops after writing a validated script. It does not build venvs, does not run the script, does not supervise jobs, and does not have a Fix loop.

## When to invoke

Trigger on any of:

- "port `<model>` to `<aws-instance>`"
- "quantize `<model>` for/on `<aws-instance>`"
- "what quant(ization) method should I use for `<model>` on `<aws-instance>`"
- "recommend a quantization for `<model>` (running on/targeting) `<aws-instance>`"

If the user gives a model but no instance (or vice versa), ask once for the missing piece before proceeding. Do not guess defaults.

## Inputs and parsing

Two regexes (lifted from `src/quant_agent/research_agent.py`):

- **Instance:** `\b([a-z]\d+[a-z\d\-]*\.(?:\d*x?large|metal))\b` (e.g. `g5.xlarge`, `p4d.24xlarge`, `g6e.12xlarge`, `p5.48xlarge`).
- **Model phrase:** the rest of the user input after stripping the instance match and the words `port`, `quantize`, `to`, `on`, `for`, `using`. Collapse whitespace.

## Phase 1 — Resolve inputs

Run these in parallel.

1. **Model alias lookup.** Read `reference/model_aliases.yaml`. Normalize the model phrase by lowercasing and replacing every run of characters not in `[a-z0-9.]` with a single space, then collapsing whitespace. Look up the normalized phrase as a key. If hit, you have `model_id` (e.g. `meta-llama/Llama-2-7b-hf`). If miss, ask the user for an exact HuggingFace id — do not invent one.

2. **Instance lookup.** Read `reference/aws_instances.yaml` and `reference/gpu_specs.yaml`. From the instance type, extract `{vram_gb, gpu_count, gpu}`. Then cross-reference `gpu` in `gpu_specs.yaml` to add `{compute_capability, gpu_arch}`. If the instance isn't in the YAML, tell the user it's unsupported and stop.

3. **HuggingFace model info.** WebFetch `https://huggingface.co/api/models/<model_id>`. Parse the JSON response and pull out:
   - `architectures` (e.g. `["LlamaForCausalLM"]`) — needed for compatibility reasoning in Research.
   - `safetensors.total` if present — the exact param count. If absent, sum `safetensors.parameters` across the entries, or fall back to asking the user for `params_b` (model parameters in billions).
   - `params_b = total_params / 1e9`. If you can't get a number, hold `params_b = None` and skip VRAM math; tell the user the recommendation will be coarser.

4. **Echo resolved inputs.** Before running Research, print one line summarizing what you resolved:

   ```
   resolved: model=<id> params_b=<n> instance=<type> vram=<g>GB gpu=<name> compute_cap=<c> arch=<a>
   ```

## Phase 2 — Research

Read `reference/methods.yaml` (the catalog, ~40 methods).

Your job is to survey **every** catalog method and produce a ranked candidate list. **You do not pick a winner — the user picks.**

### Per-method walk

For **every** id in `methods.yaml`, emit one verdict line under a `## Considered` section:

- `verdict: include` if the method plausibly supports this model's architecture **and** runs on this GPU (compute capability, kernel availability, supported bit widths) **and** fits VRAM at a supported bit width.
- `verdict: reject` otherwise.
- `reason:` one line citing the specific axis: architecture compatibility (`hf_info.architectures` vs the method's typical targets), GPU/compute-capability fit (e.g. FP8 needs Hopper sm_90, Marlin kernels need Ampere sm_80+), VRAM math (`params_b * bits / 8 * 1.4 <= vram_gb`), or calibration/QAT fit.

You **must** produce exactly one `considered` entry per catalog id. No duplicates, no omissions. Render it as a collapsible section so the table below stays the focus:

```markdown
<details>
<summary>Considered (all <N> methods)</summary>

- `gptq` — **include** — weight-only PTQ, supports Llama, fits 5.1 GB at 4-bit on 24GB A10G.
- `awq` — **include** — activation-aware weight-only, kernel-mature on Ampere sm_80+.
- `fp8` — **reject** — FP8 requires Hopper sm_90; A10G is sm_86.
- ... (one line per id)

</details>
```

### Finalists

From the `include` verdicts, pick **3-8** and emit them as a markdown table titled `## Candidates`:

| # | id | name | bits | est_vram_gb | quality | speed | needs_calib | summary |
|---|----|------|------|-------------|---------|-------|-------------|---------|
| 1 | awq | Activation-aware Weight Quantization | 4 | 5.6 | 5 | 4 | yes | 2-3 sentence why/when |

Hard rules (mirror the Pydantic validator in `src/quant_agent/schemas.py`):

- `id` MUST be a catalog id from `methods.yaml`.
- `name` and `quality`/`speed` columns MUST be copied verbatim from the catalog (`quality` and `speedup` fields).
- `bits` MUST be one of the catalog's `bits` for that id, picked so that `params_b * bits / 8 * 1.4 <= vram_gb`.
- `est_vram_gb = round(params_b * bits / 8 * 1.4, 2)`.
- `needs_calib` mirrors the catalog's `needs_calibration` field.
- `summary` is 2-3 sentences on when this method is the right fit and what it costs.

### Tradeoffs

After the table, write one paragraph titled `## Tradeoffs` comparing the finalists across the axes that matter for **this** model and GPU: quality vs speed, calibration cost, bit-width options, activation vs weight-only, kernel maturity. **No ranking** — the user ranks.

## Phase 3 — User selection

Ask the user:

> Pick a candidate by number (1-N), or `q` to quit:

If the user replies with a number in range, use that candidate as `chosen`. If `q` or anything else, stop cleanly and explain how to retry.

## Phase 4 — Adapt

You are now writing a Python script that quantizes `<model_id>` using `<chosen.name>` (`<chosen.id>`, `<chosen.bits>`-bit). Output path: `out/quantize_<safe_model>_<chosen.id>.py` where `safe_model = re.sub(r'[^a-zA-Z0-9._-]+', '_', model_id).strip('_')`. Create `out/` if it doesn't exist (`mkdir -p out`).

### Workflow — follow in order

1. **Read the README.** WebFetch `https://raw.githubusercontent.com/<owner>/<repo>/HEAD/README.md` (where `<owner>/<repo>` comes from `chosen.repo_url`). If 404, retry on `master/README.md`, then `main/README.md`. Note install steps and example usage.

2. **Clone the repo locally.** `Bash: git clone --depth 1 <chosen.repo_url> /tmp/quant-<chosen.id>/repo` (idempotent: skip if `/tmp/quant-<chosen.id>/repo/.git` exists). This is for source inspection only — you are not building a venv.

3. **Identify install steps from the README.** Examples (matching the Python agent's defaults):
   - Research repos: `["pip install -r requirements.txt", "pip install -e ."]` or `["pip install -e ."]`
   - Pip-packaged libs: `["pip install autoawq"]` (awq), `["pip install gptqmodel datasets"]` (gptq), `["pip install hqq"]` (hqq), `["pip install bitsandbytes"]` (bnb_nf4 / bnb_llm_int8)

   Always assume a baseline of `torch==2.3.1` (cu121) + `transformers` + `accelerate` + `safetensors` + `sentencepiece` is installed first. **Do NOT duplicate those in the install_steps you write into the script header.**

4. **Find the example entry point.** `Bash: ls /tmp/quant-<chosen.id>/repo/examples`, then `ls .../scripts`, then `ls .../` if needed. `Read` 1-2 most-likely entry files to see their API. **Do not run `--help`** (no venv built); read the argparse/click block in the source instead to discover the real flag names.

5. **Decide the script style and write it** with the `Write` tool to `out/quantize_<safe_model>_<chosen.id>.py`:

   **STYLE A — Standalone import** (preferred when the method has a stable Python API: autoawq, gptqmodel, hqq, bitsandbytes, or any importable module in the cloned repo). One Python file that imports the library and runs quantization in-process. Must load `<model_id>` from HF and save output to `./quantized/<chosen.id>-<safe_model>/`.

   **STYLE B — Wrapper subprocess** (preferred for research repos whose usage is `python examples/foo.py --model ...`). Python file that:
   - imports `sys, os, subprocess, pathlib`
   - reads `HF_TOKEN` from env (`HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN`) and passes it into the child env
   - computes `REPO = pathlib.Path("/tmp/quant-<chosen.id>/repo").resolve()`
   - `subprocess.run([sys.executable, "<entry_path_relative_to_repo>", "<flag_name>", "<model_id>", "<output_flag>", str(OUT)], cwd=REPO, env={**os.environ, ...}, check=True)`
   - `OUT = pathlib.Path("./quantized/<chosen.id>-<safe_model>").resolve()`
   - **EXACT flag names MUST come from the example source you read.** Do not invent flag names.

   Both styles must:
   - Quantize the real model at `<model_id>`.
   - Handle HF gated models: read `HF_TOKEN` from env and pass to every loader / tokenizer call (or `huggingface_hub.login()`), and into the child env for wrapper style.
   - Print a final line with the quantized model's on-disk size so success is visible in stdout.

6. **Prepend a header comment block** to the script with the install steps the user must run first:

   ```python
   # Generated by the `quant` Claude Code skill.
   # Method: <chosen.name> (<chosen.id>), <chosen.bits>-bit
   # Model:  <model_id>
   # Target: <instance_type> (<vram_gb>GB <gpu>, sm_<compute_capability>)
   #
   # Before running, install the method into a clean venv:
   #   python3.10 -m venv .venv-<chosen.id>
   #   source .venv-<chosen.id>/bin/activate
   #   pip install --upgrade pip wheel
   #   pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
   #   pip install transformers accelerate safetensors sentencepiece
   #   <method-specific install steps from README, e.g.: pip install autoawq>
   #
   # Then: python <this-script>
   ```

## Phase 5 — Validate

Run **stage 1** (`ast.parse`) only — this skill never builds a venv, so the dry-import stage is intentionally skipped (the Python agent skips it too when the venv is absent; see `src/quant_agent/tools/script_io.py`).

```
Bash: python3 -c "import ast,sys; ast.parse(open('out/quantize_<safe_model>_<chosen.id>.py').read())"
```

- If exit 0: validation passed, continue to handoff.
- If exit non-zero: the stderr will contain `SyntaxError: <msg> at line <n>`. Use `Edit` to fix the script and re-run validation. **Retry up to 3 times.**
- If still failing after 3 attempts: prepend `# WARNING: failed validation at stage=parse: <msg>` to the file and tell the user the script may need manual fixes.

## Phase 6 — Handoff

Print, in order:

1. The script path (`out/quantize_<safe_model>_<chosen.id>.py`).
2. The chosen method (`<chosen.id>` / `<chosen.name>`, `<chosen.bits>`-bit).
3. The exact install commands from the script header.
4. The run command: `python out/quantize_<safe_model>_<chosen.id>.py`.

Do not offer to run the script — execution is intentionally out of scope.

## Guardrails

- **Don't invent flag names.** Style B subprocesses must use flag strings copied from the cloned example's argparse/click block. If you can't find the source, fall back to Style A.
- **Don't duplicate the torch/transformers baseline** in the install-steps comment block beyond the explicit baseline lines shown above. The Python agent treats those as preinstalled.
- **Don't reference `.venvs/<method>/` paths in the script.** The Python agent uses that layout because its executor builds those venvs; this skill does not. Use `/tmp/quant-<chosen.id>/repo` for cloned-repo references and let the user activate their own venv.
- **Don't fan out RAG calls or arxiv fetches during Research.** Research is catalog-only. The `notes` field in `methods.yaml` plus your model knowledge is the grounding.
- **Don't pick a winner during Research.** The user picks.
- **Don't proceed if the model can't be resolved.** Ask for an exact HF id.

## Reference files (bundled in this skill)

- `reference/methods.yaml` — the 40-method catalog, copied verbatim from `seed/methods.yaml`. Authoritative for `id`, `repos`, `bits`, `quality`, `speedup`, `needs_calibration`, `notes`.
- `reference/model_aliases.yaml` — fuzzy model phrase → canonical HF id, copied verbatim from `seed/model_aliases.yaml`.
- `reference/aws_instances.yaml` — instance type → `{vram_gb, gpu_count, gpu}`, copied verbatim from `seed/aws_instances.yaml`.
- `reference/gpu_specs.yaml` — GPU model → `{compute_capability, gpu_arch}`, copied verbatim from `seed/gpu_specs.yaml`.

These four files are the entire grounding surface. If a fact you need is not in them and not derivable from the user's input or a single WebFetch on HuggingFace / GitHub, ask the user instead of guessing.
