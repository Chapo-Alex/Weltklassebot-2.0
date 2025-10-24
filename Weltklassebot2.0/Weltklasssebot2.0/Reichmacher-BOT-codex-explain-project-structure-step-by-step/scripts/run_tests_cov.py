"""Run the test suite with coverage, tolerating absent pytest-cov."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Sequence


def _run(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    extras = list(argv or ())
    has_pytest_cov = importlib.util.find_spec("pytest_cov") is not None
    if has_pytest_cov:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--cov=src",
            "--cov-report=xml",
            *extras,
        ]
        return _run(command)
    if importlib.util.find_spec("coverage") is None:
        sys.stderr.write(
            "coverage package is required when pytest-cov is unavailable.\n",
        )
        return 1
    test_command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "-m",
        "pytest",
        "-q",
        *extras,
    ]
    result = _run(test_command)
    xml_code = _run([sys.executable, "-m", "coverage", "xml"])
    if result == 0:
        return xml_code
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
