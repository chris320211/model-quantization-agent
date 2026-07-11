from typer.testing import CliRunner
from typer.main import get_command

from quant_agent.cli import app


def test_ask_help_documents_execution_and_fallback_policy():
    result = CliRunner().invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--fallback-candidates" in result.stdout
    command = get_command(app).commands["ask"]
    options = {opt for param in command.params for opt in getattr(param, "opts", [])}
    assert "--allow-unsafe-host-execution" in options
    assert "--unsafe-host" in options


def test_adapt_help_documents_execution_policy():
    result = CliRunner().invoke(app, ["adapt", "--help"])
    assert result.exit_code == 0
    command = get_command(app).commands["adapt"]
    options = {opt for param in command.params for opt in getattr(param, "opts", [])}
    assert "--allow-unsafe-host-execution" in options
    assert "--unsafe-host" in options
