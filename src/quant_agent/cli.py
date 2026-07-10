from __future__ import annotations

import typer

from . import adapt_only as adapt_only_module
from . import executor as executor_module
from . import orchestrator as orchestrator_module
from . import setup_cmd as setup_module

app = typer.Typer(
    add_completion=False,
    help="LangChain quantization-porting agent for HuggingFace LLMs (on-box EC2 executor).",
    no_args_is_help=True,
)

jobs_app = typer.Typer(help="Inspect background quantization jobs.")
app.add_typer(jobs_app, name="jobs")


@app.command("setup")
def setup_cmd(
    force: bool = typer.Option(False, "--force", help="Overwrite an existing .env."),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Make a tiny live call to verify the key."
    ),
    no_optional: bool = typer.Option(
        False, "--no-optional", help="Skip prompts for GITHUB_TOKEN / HUGGINGFACE_HUB_TOKEN."
    ),
) -> None:
    """Interactively write .env with your API keys (hidden input, chmod 600)."""
    raise typer.Exit(setup_module.run(force=force, validate=validate, no_optional=no_optional))


@app.command("ask")
def ask_cmd(
    request: str = typer.Argument(..., help="Natural-language quantization request."),
    dry: bool = typer.Option(
        False, "--dry", help="Stop after writing the validated script; skip execution."
    ),
    max_repairs: int = typer.Option(
        3,
        "--max-repairs",
        help="Runtime-failure repair attempts on the chosen method (0 disables).",
    ),
    max_adapt_retries: int = typer.Option(
        2,
        "--max-adapt-retries",
        help="Same-method adapt-time retries before falling back to the next candidate.",
    ),
    tune: bool = typer.Option(
        False,
        "--tune",
        help="After baseline succeeds, prompt to enter the closed-loop hyperparameter tuner.",
    ),
    auto_tune: bool = typer.Option(
        False,
        "--auto-tune",
        help="Imply --tune and skip the 'Tune further?' prompt (non-interactive).",
    ),
    max_tune_iter: int = typer.Option(
        5,
        "--max-tune-iter",
        help="Hard cap on tune iterations (including the baseline).",
    ),
    stagnate_after: int = typer.Option(
        2,
        "--stagnate-after",
        help="Stop tuning after this many consecutive non-improvements.",
    ),
) -> None:
    """Research → pick → Adapt → execute → (optional) closed-loop tuner."""
    typer.echo(
        orchestrator_module.run(
            request,
            dry=dry,
            max_repairs=max_repairs,
            max_adapt_retries=max_adapt_retries,
            tune=tune,
            auto_tune=auto_tune,
            max_tune_iter=max_tune_iter,
            stagnate_after=stagnate_after,
        )
    )


@app.command("adapt")
def adapt_cmd(
    method_id: str = typer.Argument(..., help="Catalog id from seed/methods.yaml, e.g. 'flatquant'."),
    model_id: str = typer.Argument(..., help="Canonical HuggingFace model id, e.g. 'meta-llama/Llama-2-7b-hf'."),
    bits: int = typer.Option(None, "--bits", help="Target bit-width (default: first value in the method's 'bits' list)."),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Allow executing the model's custom modeling code (auto_map models). Off by default.",
    ),
) -> None:
    """Skip Research/selection and drive the Adapt agent directly against a known (method, model) pair."""
    script_path, _ = adapt_only_module.run(
        method_id=method_id, model_id=model_id, bits=bits, trust_remote_code=trust_remote_code
    )
    typer.echo(f"Script written: {script_path}")


@jobs_app.command("list")
def jobs_list() -> None:
    metas = executor_module.list_jobs()
    if not metas:
        typer.echo("No jobs.")
        return
    for m in metas:
        typer.echo(
            f"{m.job_id}  {m.status:<10} {m.method_id:<8} {m.model_id}  pid={m.pid}"
        )


@jobs_app.command("status")
def jobs_status(job_id: str) -> None:
    meta = executor_module.refresh_status(job_id)
    typer.echo(meta.to_json())


@jobs_app.command("logs")
def jobs_logs(job_id: str, n: int = typer.Option(80, "-n", help="Lines per stream")) -> None:
    logs = executor_module.tail(job_id, n_lines=n)
    typer.echo("=== stdout ===")
    typer.echo(logs["stdout.log"])
    typer.echo("=== stderr ===")
    typer.echo(logs["stderr.log"])


@jobs_app.command("kill")
def jobs_kill(job_id: str) -> None:
    meta = executor_module.kill(job_id)
    typer.echo(meta.to_json())


@app.callback()
def default(ctx: typer.Context) -> None:
    """LangChain quantization-porting agent. Use a subcommand (e.g. `ask`, `setup`)."""
    return


if __name__ == "__main__":
    app()
