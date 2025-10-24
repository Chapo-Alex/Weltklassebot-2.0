"""Validate optional import paths for :mod:`strategy.breakout_bias`."""

from __future__ import annotations

import importlib
import sys
import types


def test_breakout_bias_configdict(monkeypatch) -> None:
    """Ensure ConfigDict aliasing executes when pydantic v2 is available."""

    original_module = sys.modules.get("strategy.breakout_bias")
    original_pydantic = sys.modules.get("pydantic")
    if "strategy.breakout_bias" in sys.modules:
        del sys.modules["strategy.breakout_bias"]

    fake = types.ModuleType("pydantic")

    class FakeBaseModel:  # pragma: no cover - executed via reload
        def __init__(self, **data):  # type: ignore[no-untyped-def]
            for key, value in data.items():
                setattr(self, key, value)

    def fake_field(*, default=None, default_factory=None, alias=None):  # type: ignore[no-untyped-def]
        if default_factory is not None:
            return default_factory()
        return default

    def fake_config_dict(**kwargs):  # type: ignore[no-untyped-def]
        return dict(**kwargs)

    fake.BaseModel = FakeBaseModel
    fake.Field = fake_field
    fake.ConfigDict = fake_config_dict
    monkeypatch.setitem(sys.modules, "pydantic", fake)

    module = importlib.import_module("strategy.breakout_bias")
    assert hasattr(module.StrategyConfig, "model_config")
    assert module.StrategyConfig.model_config["arbitrary_types_allowed"] is True

    # Restore original modules.
    monkeypatch.delitem(sys.modules, "strategy.breakout_bias", raising=False)
    if original_module is not None:
        sys.modules["strategy.breakout_bias"] = original_module
    else:  # pragma: no cover - defensive path
        importlib.import_module("strategy.breakout_bias")

    if original_pydantic is not None:
        sys.modules["pydantic"] = original_pydantic
    else:
        monkeypatch.delitem(sys.modules, "pydantic", raising=False)
