from __future__ import annotations

import typer

from . import agent as agent_module
from . import ingest as ingest_module

app = typer.Typer(
    add_completion=False,
    help="LangChain quantization-porting agent for HuggingFace LLMs.",
    no_args_is_help=True,
)


@app.command("ingest")
def ingest_cmd() -> None:
    """Build or refresh the local Chroma index from seed/methods.yaml."""
    n = ingest_module.ingest_all()
    typer.echo(f"Added {n} new chunks.")


@app.command("ask")
def ask_cmd(request: str = typer.Argument(..., help="Natural-language quantization request.")) -> None:
    """Ask the agent to port a model and generate a quantization script."""
    typer.echo(agent_module.run(request))


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context, request: str | None = typer.Argument(None)) -> None:
    """Default: treat bare invocation as `ask`."""
    if ctx.invoked_subcommand is not None:
        return
    if not request:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)
    typer.echo(agent_module.run(request))


if __name__ == "__main__":
    app()
