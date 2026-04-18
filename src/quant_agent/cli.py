from __future__ import annotations

from dataclasses import asdict

import typer

from . import agent as agent_module
from . import executor as executor_module
from . import ingest as ingest_module

app = typer.Typer(
    add_completion=False,
    help="LangChain quantization-porting agent for HuggingFace LLMs (on-box EC2 executor).",
    no_args_is_help=True,
)

jobs_app = typer.Typer(help="Inspect background quantization jobs.")
app.add_typer(jobs_app, name="jobs")


@app.command("ingest")
def ingest_cmd() -> None:
    """Build or refresh the local Chroma index from seed/methods.yaml."""
    n = ingest_module.ingest_all()
    typer.echo(f"Added {n} new chunks.")


@app.command("ask")
def ask_cmd(request: str = typer.Argument(..., help="Natural-language quantization request.")) -> None:
    """Ask the agent to recommend, optionally generate a script, or execute a port."""
    typer.echo(agent_module.run(request))


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


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context, request: str | None = typer.Argument(None)) -> None:
    """Default: treat bare invocation as `ask`."""
    if ctx.invoked_subcommand is not None:
        return
    if not request:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)
    typer.echo(agent_module.run(request))


# asdict kept in import scope for future structured output
_ = asdict


if __name__ == "__main__":
    app()
