import json

from cli import app
from typer.testing import CliRunner


def test_backtest_synthetic_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["backtest", "--seed", "123"], catch_exceptions=False)
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip())
    assert payload["sha256"]
    assert payload["lines"] > 0
