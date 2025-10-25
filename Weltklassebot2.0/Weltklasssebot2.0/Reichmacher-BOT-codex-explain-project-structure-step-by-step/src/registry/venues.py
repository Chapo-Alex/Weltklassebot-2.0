"""Registry providing venue-specific fee and slippage profiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from execution.fees import FeeModel, FlatFee, TieredFee
from execution.slippage import LinearSlippage, SlippageModel, SquareRootSlippage

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class VenueProfile:
    """Concrete fee, slippage, and constraint models for a venue/symbol pair."""

    fee_model: FeeModel
    slippage_model: SlippageModel
    tick_size: float
    min_qty: float


def _default_profile() -> VenueProfile:
    return VenueProfile(
        fee_model=FlatFee(bps=0.0),
        slippage_model=LinearSlippage(),
        tick_size=1e-8,
        min_qty=1e-6,
    )


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "venues.yaml"


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        _LOG.info("PyYAML not available, skipping venue overrides at %s", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-untyped-call]
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        _LOG.warning("Invalid venue configuration format in %s", path)
        return {}
    return data


def _positive_value(
    value: Any,
    default: float,
    *,
    field: str,
    scope: str,
) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _LOG.warning("Invalid %s '%s' for %s; using default %.3g", field, value, scope, default)
        return default
    if parsed <= 0.0:
        _LOG.warning("Non-positive %s %.3g for %s; using default %.3g", field, parsed, scope, default)
        return default
    return parsed


def _build_fee_model(spec: Mapping[str, Any] | None) -> FeeModel:
    if not spec:
        return FlatFee(bps=0.0)
    model_type = str(spec.get("type", "flat")).lower()
    if model_type == "flat":
        bps = float(spec.get("bps", 0.0))
        taker = spec.get("taker_bps")
        taker_bps = float(taker) if taker is not None else None
        return FlatFee(bps=bps, taker_bps_override=taker_bps)
    if model_type == "tiered":
        raw_tiers = spec.get("tiers", [])
        if not isinstance(raw_tiers, list) or not raw_tiers:
            _LOG.warning("Tiered fee model requires non-empty tiers; falling back to flat")
            return FlatFee(bps=0.0)
        tiers: list[tuple[float, float]] = []
        for entry in raw_tiers:
            try:
                threshold, bps = entry
            except (TypeError, ValueError):
                _LOG.warning("Invalid tier entry %s; skipping", entry)
                continue
            tiers.append((float(threshold), float(bps)))
        cap = spec.get("cap_abs")
        cap_abs = float(cap) if cap is not None else None
        if not tiers:
            return FlatFee(bps=0.0)
        return TieredFee(tiers=tiers, cap_abs=cap_abs)
    _LOG.warning("Unknown fee model type '%s'; defaulting to flat", model_type)
    return FlatFee(bps=0.0)


def _build_slippage_model(spec: Mapping[str, Any] | None) -> SlippageModel:
    if not spec:
        return LinearSlippage()
    model_type = str(spec.get("type", "linear")).lower()
    seed = spec.get("seed")
    seed_value = int(seed) if seed is not None else None
    if model_type in {"linear", "impact_linear"}:
        bps = float(spec.get("bps_per_notional", 0.0))
        eta = float(spec.get("maker_queue_eta", 1.0))
        model = LinearSlippage(bps_per_notional=bps, maker_queue_eta=eta, seed=seed_value)
        return model
    if model_type in {"sqrt", "square_root", "impact_sqrt"}:
        k = float(spec.get("k", 0.0))
        eta = float(spec.get("maker_queue_eta", 1.0))
        model = SquareRootSlippage(k=k, maker_queue_eta=eta, seed=seed_value)
        return model
    _LOG.warning("Unknown slippage model type '%s'; defaulting to linear", model_type)
    return LinearSlippage(seed=seed_value)


def resolve_models(venue: str, symbol: str | None = None) -> VenueProfile:
    """Return fee and slippage models for the requested venue/symbol."""

    data = _load_yaml(_config_path())
    venues = data.get("venues") if isinstance(data, Mapping) else None
    if not isinstance(venues, Mapping):
        return _default_profile()
    entry = venues.get(venue, {})
    fee_spec = entry.get("fee") if isinstance(entry, Mapping) else None
    slippage_spec = entry.get("slippage") if isinstance(entry, Mapping) else None
    tick_size = 1e-8
    min_qty = 1e-6
    if isinstance(entry, Mapping):
        tick_size = _positive_value(
            entry.get("tick_size"),
            tick_size,
            field="tick_size",
            scope=f"venue:{venue}",
        )
        min_qty = _positive_value(
            entry.get("min_qty"),
            min_qty,
            field="min_qty",
            scope=f"venue:{venue}",
        )
    if symbol and isinstance(entry, Mapping):
        symbols = entry.get("symbols")
        if isinstance(symbols, Mapping):
            symbol_entry = symbols.get(symbol)
            if isinstance(symbol_entry, Mapping):
                fee_spec = symbol_entry.get("fee", fee_spec)
                slippage_spec = symbol_entry.get("slippage", slippage_spec)
                tick_size = _positive_value(
                    symbol_entry.get("tick_size"),
                    tick_size,
                    field="tick_size",
                    scope=f"{venue}:{symbol}",
                )
                min_qty = _positive_value(
                    symbol_entry.get("min_qty"),
                    min_qty,
                    field="min_qty",
                    scope=f"{venue}:{symbol}",
                )
    fee_model = _build_fee_model(fee_spec)
    slippage_model = _build_slippage_model(slippage_spec)
    return VenueProfile(
        fee_model=fee_model,
        slippage_model=slippage_model,
        tick_size=tick_size,
        min_qty=min_qty,
    )


__all__ = ["VenueProfile", "resolve_models"]
