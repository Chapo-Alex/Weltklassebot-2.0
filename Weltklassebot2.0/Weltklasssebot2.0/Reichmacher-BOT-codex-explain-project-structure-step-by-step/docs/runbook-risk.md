# Risk Manager Runbook

## Zustände und Übergänge

Der `portfolio.risk.RiskManagerV2` hält eine endliche State-Machine mit drei klar
getrennten Betriebsmodi:

| Zustand   | Wert | Bedeutung |
|-----------|------|-----------|
| `RUNNING` | `0`  | Normalbetrieb. Orders werden nur anhand der konfigurierten Limits (Drawdown, Notional, Trades pro Tag) geprüft. |
| `COOLDOWN` | `1` | Zeitlich begrenzter Handelsstopp. Entsteht automatisch, wenn das Tages-Handelslimit erreicht wurde und `cooldown_minutes > 0` ist oder manuell via Admin-API. |
| `HALTED` | `2` | Harte Sperre, ausgelöst durch maximale Drawdown-Verletzung oder manuell via Admin-API. |

Die Transitionen werden jedes Mal evaluiert, wenn `RiskManagerV2.transition`
aufgerufen wird. Typische automatisierte Übergänge:

- `RUNNING → COOLDOWN`: Tages-Tradelimit erreicht und ein Cooldown-Fenster ist
  konfiguriert.
- `COOLDOWN → RUNNING`: Das Cooldown-Ende (`ctx.now >= cooldown_until`) ist
  erreicht.
- `RUNNING → HALTED`: Der beobachtete Drawdown (`RiskContext.drawdown`) erreicht
  oder überschreitet `RiskParameters.max_drawdown`.
- `HALTED → RUNNING`: Nur über einen manuellen Admin-Eingriff (siehe unten).

Jeder Zustandswechsel aktualisiert Prometheus-Metriken (`core.metrics.RISK_STATE`
und Counter `RISK_DENIALS`) und wird optional auditierbar persistiert, wenn eine
State-Store-Konfiguration aktiv ist.

## Admin-Befehle

Verwenden Sie die Helfer aus `portfolio.risk_admin`, um den Zustand gezielt zu
steuern. Alle Operationen schreiben sowohl den persistierten Zustand als auch
Audit-Einträge (JSON Lines).

```python
from pathlib import Path
from portfolio.risk_admin import force_cooldown, halt, resume, get_state
from portfolio.risk_state_store import JsonlStateStore

store = JsonlStateStore(Path("/var/lib/weltklasse/risk"))

# Aktuellen Status anzeigen
print(get_state(store))

# 15 Minuten Cooldown setzen (z. B. bei Volatilitätsspitze)
force_cooldown(store, minutes=15, actor="sre@desk", reason="volatility_spike")

# Hard Stop nach Incident
halt(store, actor="sre@desk", reason="risk_limit_breach")

# Trading wieder erlauben
resume(store, actor="sre@desk", reason="manual_clearance")
```

Alle Funktionen sind deterministisch und idempotent. Ein erneuter Aufruf mit
denselben Parametern überschreibt den Zustand und hängt einen zusätzlichen
Audit-Eintrag an.

## Persistenzpfade und Wiederanlauf

Der `JsonlStateStore` legt Dateien unterhalb des konfigurierten Verzeichnisses
an (Standardnamen `risk_state.json` und `risk_audit.jsonl`). Eigenschaften:

- **Atomare Writes**: `save_state` schreibt zuerst in eine temporäre Datei und
  ersetzt anschließend die bestehende State-Datei per `os.replace`. Dadurch ist
  der Zustand selbst bei Abstürzen konsistent.
- **Audit-Trail**: `append_audit` hängt eine JSON-Zeile pro Aktion an. Jede Zeile
  enthält mindestens `ts`, `actor`, `action` und `reason`.
- **Engine-Integration**: Wird `BacktestConfig.risk_store_dir` gesetzt, lädt die
  Engine den zuletzt persistierten Zustand beim Start und schreibt bei jedem
  Transition-Callback neue Snapshots/Audit-Einträge.

### Beispiel: Persistenz überprüfen

```bash
$ jq '.' /var/lib/weltklasse/risk/risk_state.json
{
  "state": "COOLDOWN",
  "since": 1700000000.0,
  "meta": {"cooldown_minutes": 15}
}

$ tail -n5 /var/lib/weltklasse/risk/risk_audit.jsonl
{"ts":1700000000.0,"actor":"sre@desk","action":"cooldown","reason":"volatility_spike"}
```

### Crash-Recovery

Bei einem Neustart liest die Engine automatisch `risk_state.json`. Befindet sich
der gespeicherte Zustand in `COOLDOWN` oder `HALTED`, bleibt der Handelsfluss
blockiert, bis ein Admin über die oben genannten Befehle einen neuen Zustand
persistiert. Die Audit-Datei ermöglicht eine rückwirkende Analyse aller manuellen
Eingriffe.
