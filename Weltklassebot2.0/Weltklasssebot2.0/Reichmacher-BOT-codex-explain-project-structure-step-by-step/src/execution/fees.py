"""Fee model abstractions used by execution venues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class FeeModel(Protocol):
    """Protocol describing maker/taker fee computations."""

    def maker_bps(self, notional: float) -> float:
        """Return the maker fee in basis points for the provided notional."""

    def taker_bps(self, notional: float) -> float:
        """Return the taker fee in basis points for the provided notional."""

    def apply(self, notional: float, is_maker: bool) -> float:
        """Return the absolute fee for the notional with maker/taker selection."""


def _ensure_non_negative(value: float, *, name: str) -> None:
    if value < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


@dataclass(slots=True)
class FlatFee(FeeModel):
    """Flat maker/taker fee expressed in basis points."""

    bps: float
    taker_bps_override: float | None = None

    def __post_init__(self) -> None:
        _ensure_non_negative(self.bps, name="bps")
        if self.taker_bps_override is not None:
            _ensure_non_negative(self.taker_bps_override, name="taker_bps")

    def maker_bps(self, notional: float) -> float:  # noqa: D401 - short alias
        del notional
        return self.bps

    def taker_bps(self, notional: float) -> float:  # noqa: D401 - short alias
        del notional
        if self.taker_bps_override is not None:
            return self.taker_bps_override
        return self.bps

    def apply(self, notional: float, is_maker: bool) -> float:
        rate_bps = self.maker_bps(notional) if is_maker else self.taker_bps(notional)
        return abs(notional) * (rate_bps / 10_000.0)


@dataclass(slots=True)
class TieredFee(FeeModel):
    """Tiered fee schedule based on notional thresholds."""

    tiers: list[tuple[float, float]]
    cap_abs: float | None = None

    def __post_init__(self) -> None:
        if not self.tiers:
            msg = "tiers must not be empty"
            raise ValueError(msg)
        ordered = sorted(self.tiers, key=lambda item: item[0])
        self.tiers[:] = ordered
        for threshold, bps in self.tiers:
            _ensure_non_negative(threshold, name="threshold")
            _ensure_non_negative(bps, name="bps")
        if self.cap_abs is not None:
            _ensure_non_negative(self.cap_abs, name="cap_abs")

    def _lookup(self, notional: float) -> float:
        absolute = abs(notional)
        candidate_bps = self.tiers[0][1]
        for threshold, bps in self.tiers:
            candidate_bps = bps
            if absolute <= threshold:
                break
        return candidate_bps

    def maker_bps(self, notional: float) -> float:
        return self._lookup(notional)

    def taker_bps(self, notional: float) -> float:
        return self._lookup(notional)

    def apply(self, notional: float, is_maker: bool) -> float:
        del is_maker
        fee = abs(notional) * (self._lookup(notional) / 10_000.0)
        if self.cap_abs is not None:
            return min(fee, self.cap_abs)
        return fee


__all__ = ["FeeModel", "FlatFee", "TieredFee"]
