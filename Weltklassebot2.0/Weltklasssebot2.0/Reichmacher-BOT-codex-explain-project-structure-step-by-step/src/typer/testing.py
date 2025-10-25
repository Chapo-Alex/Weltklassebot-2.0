"""Testing helpers compatible with Typer's CliRunner."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Sequence

from . import Exit


@dataclass
class Result:
    exit_code: int
    stdout: str
    stderr: str
    exception: Exception | None


class CliRunner:
    """Very small stand-in for Typer's CliRunner."""

    def invoke(
        self,
        app: Any,
        args: Sequence[str] | None = None,
        *,
        catch_exceptions: bool = True,
    ) -> Result:
        argv = list(args or [])
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exception: Exception | None = None
        exit_code = 0

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                if hasattr(app, "_run"):
                    app._run(argv)  # type: ignore[attr-defined]
                else:
                    app(*argv)
            except Exit as exc:
                exit_code = exc.code
            except Exception as exc:  # pragma: no cover - mirrors Typer behaviour
                if catch_exceptions:
                    exit_code = 1
                    exception = exc
                else:
                    raise
        return Result(
            exit_code=exit_code,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            exception=exception,
        )
