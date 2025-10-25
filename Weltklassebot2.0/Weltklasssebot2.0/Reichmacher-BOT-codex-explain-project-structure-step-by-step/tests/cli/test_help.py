from cli import app
from typer.testing import CliRunner


def test_cli_help_lists_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "backtest" in result.stdout
    assert "walkforward" in result.stdout
    assert "--seed" in result.stdout
