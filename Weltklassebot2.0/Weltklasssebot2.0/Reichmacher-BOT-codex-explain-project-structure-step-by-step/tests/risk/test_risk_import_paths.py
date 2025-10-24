"""Validate import-time branches for :mod:`portfolio.risk`."""

from __future__ import annotations

import importlib
import sys
import types


def test_risk_parameters_configdict_path(monkeypatch) -> None:
    """Ensure the ConfigDict branch is executed when pydantic v2 is present."""

    original_risk = sys.modules.get("portfolio.risk")
    original_pydantic = sys.modules.get("pydantic")
    if "portfolio.risk" in sys.modules:
        del sys.modules["portfolio.risk"]

    fake = types.ModuleType("pydantic")

    class FakeBaseModel:  # pragma: no cover - executed via reload
        def __init__(self, **data):  # type: ignore[no-untyped-def]
            for key, value in data.items():
                setattr(self, key, value)

    def fake_config_dict(**kwargs):  # type: ignore[no-untyped-def]
        return dict(**kwargs)

    fake.BaseModel = FakeBaseModel
    fake.ConfigDict = fake_config_dict
    monkeypatch.setitem(sys.modules, "pydantic", fake)

    module = importlib.import_module("portfolio.risk")
    assert hasattr(module.RiskParameters, "model_config")
    assert module.RiskParameters.model_config["frozen"] is True

    # Restore original modules to avoid polluting subsequent tests.
    monkeypatch.delitem(sys.modules, "portfolio.risk", raising=False)
    if original_risk is not None:
        sys.modules["portfolio.risk"] = original_risk
    else:  # pragma: no cover - defensive path
        importlib.import_module("portfolio.risk")

    if original_pydantic is not None:
        sys.modules["pydantic"] = original_pydantic
    else:
        monkeypatch.delitem(sys.modules, "pydantic", raising=False)
