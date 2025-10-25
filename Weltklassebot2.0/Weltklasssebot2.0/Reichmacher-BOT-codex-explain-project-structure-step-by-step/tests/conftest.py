# ruff: noqa: I001

from __future__ import annotations

import ast
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

# --- begin deterministic setup ---
import os  # noqa: I001
import random  # noqa: I001

os.environ.setdefault("PYTHONHASHSEED", "0")

random.seed(1337)
try:
    import numpy as _np  # type: ignore[import-not-found]
    _np.random.seed(1337)  # type: ignore[operator]
except Exception:
    pass

try:
    from hypothesis import settings

    settings.register_profile("ci", deadline=None, max_examples=15, derandomize=True)
    settings.load_profile("ci")
except Exception:
    pass
# --- end deterministic setup ---

if TYPE_CHECKING:
    from core.events import CandleEvent
    from portfolio.accounting import FeeModel
    from portfolio.risk import RiskParameters

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    _np = None

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11 interpreters
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(slots=True)
class _CoverageTarget:
    label: str
    path: Path

    def iter_files(self) -> Iterable[Path]:
        if self.path.is_file():
            yield self.path
            return
        yield from self.path.rglob("*.py")

    def matches(self, filename: Path) -> bool:
        if self.path.is_file():
            return filename == self.path
        try:
            return filename.is_relative_to(self.path)
        except ValueError:  # pragma: no cover - older Python fallback
            return str(filename).startswith(str(self.path))


class _CoverageTracer:
    def __init__(self, targets: list[_CoverageTarget], threshold: float) -> None:
        self._targets = targets
        self._threshold = threshold
        self._executed: dict[Path, set[int]] = defaultdict(set)
        self._tracing = False

    @staticmethod
    def _statement_lines(path: Path) -> set[int]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines: set[int] = set()
        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", None)
            if lineno is not None:
                lines.add(int(lineno))
        return lines

    def _trace(self, frame, event, arg):  # type: ignore[no-untyped-def]
        if event != "line":
            return self._trace
        filename = Path(frame.f_code.co_filename).resolve()
        for target in self._targets:
            if target.matches(filename):
                self._executed[filename].add(frame.f_lineno)
                break
        return self._trace

    def start(self) -> None:
        if self._tracing:
            return
        self._tracing = True
        sys.settrace(self._trace)
        threading.settrace(self._trace)

    def stop(self) -> None:
        if not self._tracing:
            return
        sys.settrace(None)
        threading.settrace(None)
        self._tracing = False

    def report(self) -> tuple[dict[str, float], float]:
        results: dict[str, float] = {}
        for target in self._targets:
            total = 0
            covered = 0
            for file in target.iter_files():
                statements = self._statement_lines(file)
                total += len(statements)
                executed = self._executed.get(file.resolve(), set())
                covered += len(statements & executed)
            ratio = 1.0 if total == 0 else covered / total
            results[target.label] = ratio
        return results, self._threshold


def _load_targets(config: pytest.Config) -> tuple[list[_CoverageTarget], float]:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = project_root / "pyproject.toml"
    threshold = 0.85
    target_paths = [
        "src/core",
        "src/portfolio",
        "src/portfolio/risk.py",
        "src/strategy",
    ]
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        reichmacher = data.get("tool", {}).get("reichmacher", {})
        threshold = float(reichmacher.get("coverage_threshold", threshold))
        target_paths = reichmacher.get("coverage_targets", target_paths)
    targets = []
    for raw_path in target_paths:
        path = (project_root / raw_path).resolve()
        raw = Path(raw_path)
        label = raw.stem if raw.suffix else raw.name
        targets.append(_CoverageTarget(label=label, path=path))
    return targets, threshold


def _set_global_seed(seed: int = 1337) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed)


def _configure_benchmark(config: pytest.Config) -> None:
    try:
        import pytest_benchmark.plugin  # noqa: F401
    except ImportError:
        return

    _set_global_seed()

    option = getattr(config, "option", None)
    if option is None:
        return

    min_rounds = getattr(option, "benchmark_min_rounds", None)
    try:
        rounds_value = int(min_rounds) if min_rounds is not None else 0
    except (TypeError, ValueError):
        rounds_value = 0
    if rounds_value < 5:
        option.benchmark_min_rounds = 5  # type: ignore[attr-defined]

    if hasattr(option, "benchmark_disable_gc"):
        option.benchmark_disable_gc = False
    if hasattr(option, "benchmark_timer"):
        option.benchmark_timer = "time.perf_counter"
    if hasattr(option, "benchmark_random_order"):
        option.benchmark_random_order = False
    if hasattr(option, "benchmark_random_base"):
        option.benchmark_random_base = 1337


def pytest_configure(config: pytest.Config) -> None:
    _set_global_seed()
    _configure_benchmark(config)
    targets, threshold = _load_targets(config)
    tracer = _CoverageTracer(targets, threshold)
    tracer.start()
    config._coverage_tracer = tracer  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # type: ignore[override]
    tracer: _CoverageTracer | None = getattr(session.config, "_coverage_tracer", None)
    if tracer is None:
        return
    tracer.stop()
    results, threshold = tracer.report()
    session.config._coverage_results = results  # type: ignore[attr-defined]
    failing = [label for label, ratio in results.items() if ratio < threshold]
    if failing:
        session.config._coverage_failed = failing  # type: ignore[attr-defined]
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus):  # type: ignore[no-untyped-def]
    tracer: _CoverageTracer | None = getattr(terminalreporter.config, "_coverage_tracer", None)
    if tracer is None:
        return
    results: dict[str, float] = getattr(terminalreporter.config, "_coverage_results", {})
    threshold = tracer._threshold
    terminalreporter.write_sep("-", f"coverage summary (threshold {threshold * 100:.0f}%)")
    for label, ratio in sorted(results.items()):
        terminalreporter.write_line(f"{label:>10}: {ratio * 100:5.1f}%")
    failing = getattr(terminalreporter.config, "_coverage_failed", [])
    if failing:
        terminalreporter.write_line(
            f"Coverage below threshold for: {', '.join(failing)}",
            yellow=True,
        )


class _FixedClock:
    """Deterministic monotonic clock used to emulate wall-clock time."""

    __slots__ = ("_current", "_step")

    def __init__(self, start: datetime, step: timedelta) -> None:
        self._current = start
        self._step = step

    @property
    def current(self) -> datetime:
        return self._current

    def now(self) -> datetime:
        value = self._current
        self._current = self._current + self._step
        return value

    def advance(
        self,
        *,
        delta: timedelta | None = None,
        seconds: float = 0.0,
        minutes: float = 0.0,
    ) -> datetime:
        if delta is None:
            delta = timedelta(seconds=seconds, minutes=minutes)
        self._current = self._current + delta
        return self._current


@pytest.fixture(scope="session")
def global_seed() -> int:
    """Expose the deterministic seed used across the test-suite."""

    return 1337


@pytest.fixture
def fixed_clock() -> _FixedClock:
    """Provide a monotonic clock starting at midnight UTC."""

    start = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
    return _FixedClock(start=start, step=timedelta(minutes=1))


@pytest.fixture
def rng() -> Callable[[int], Any]:
    """Factory returning seeded RNG instances with optional NumPy support."""

    if _np is not None:
        def _factory(seed: int = 1337) -> Any:
            return _np.random.default_rng(seed)
    else:
        class _FallbackRNG:
            def __init__(self, seed: int) -> None:
                self._random = random.Random(seed)

            def uniform(self, low: float, high: float) -> float:
                return self._random.uniform(low, high)

        def _factory(seed: int = 1337) -> Any:
            return _FallbackRNG(seed)

    return _factory


@pytest.fixture
def tiny_candles(rng: Callable[[int], Any]) -> Callable[..., list[CandleEvent]]:
    """Generate a deterministic candle sequence with a breakout pattern."""

    from core.events import CandleEvent

    def _coerce_start(value: datetime | str | None) -> datetime:
        if value is None:
            return datetime(2024, 1, 1, tzinfo=UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        return value.astimezone(UTC)

    def _factory(
        symbol: str = "BTCUSDT",
        *,
        n: int = 60,
        pattern: str = "breakout",
        start: datetime | str | None = None,
        interval_minutes: int = 1,
        base: float = 100.0,
    ) -> list[CandleEvent]:
        if n <= 0:
            return []
        if pattern != "breakout":  # pragma: no cover - unsupported pattern guard
            raise ValueError(f"unsupported pattern: {pattern}")

        generator: Any = rng()
        origin = _coerce_start(start)
        step = timedelta(minutes=interval_minutes)
        pivot = max(1, min(n // 2, n - 1))
        level = base
        candles: list[CandleEvent] = []

        for index in range(n):
            if index < pivot:
                deviation = float(generator.uniform(-0.6, 0.6))
                level = base + deviation
            elif index == pivot:
                level = base + 5.0
            else:
                increment = float(generator.uniform(0.45, 0.8))
                level += increment

            close_noise = float(generator.uniform(-0.05, 0.05))
            close_price = max(level + close_noise, 0.05)
            if index < pivot:
                open_center = level + float(generator.uniform(-0.12, 0.12))
            else:
                open_center = close_price - float(generator.uniform(0.03, 0.12))
            open_price = max(open_center, 0.05)
            high = max(open_price, close_price) + float(generator.uniform(0.05, 0.15))
            low = max(min(open_price, close_price) - float(generator.uniform(0.05, 0.15)), 0.01)
            start_ts = origin + step * index
            end_ts = start_ts + step
            volume = 20.0 + index * 1.5

            candles.append(
                CandleEvent(
                    symbol=symbol,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                    start=start_ts,
                    end=end_ts,
                )
            )

        return candles

    return _factory


@pytest.fixture
def risk_params_default() -> RiskParameters:
    """Return compact risk limits suitable for unit tests."""

    from portfolio.risk import RiskParameters

    return RiskParameters(
        max_drawdown=0.05,
        max_notional=10_000.0,
        max_trades_per_day=3,
        cooldown_minutes=5.0,
    )


@pytest.fixture
def fee_model_dummy() -> FeeModel:
    """Provide a deterministic flat-fee model for accounting tests."""

    from portfolio.accounting import FeeModel

    class _DummyFeeModel:
        def fee(self, qty: float, price: float, taker: bool) -> float:  # noqa: D401
            del qty, price, taker
            return 0.25

    return cast(FeeModel, _DummyFeeModel())
