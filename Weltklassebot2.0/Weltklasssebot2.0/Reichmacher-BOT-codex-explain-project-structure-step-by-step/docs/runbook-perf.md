# Performance Runbook

## Zweck

`scripts/bench_backtester.py` misst die End-to-End-Laufzeit des Backtesters auf
deterministischen Candles und liefert Kennzahlen für Durchsatz, Stage-Latenz und
Speicherspitzen. Die Ergebnisse sind Grundlage für das Perf-Budget und werden im
CI-Job `perf` validiert.

## Script-Aufruf

```bash
PYTHONPATH=src python scripts/bench_backtester.py \
  --candles 50000 \
  --seed 1337 \
  --repeats 2 \
  --symbol BTCUSDT
```

Die Ausgabe ist ein JSON-Objekt auf stdout, z. B.:

```json
{
  "candles": 50000,
  "repeats": 2,
  "seed": 1337,
  "candles_per_sec": {"min": 1025.4, "median": 1048.2, "p95": 1056.7},
  "p95_stage_latency_ms": 21.3,
  "peak_rss_mb": 182.4
}
```

### Parameter

- `--candles`: Anzahl synthetischer Kerzen (Standard: 200000).
- `--seed`: RNG-Seed für Candle-Generation und Engine (Standard: 1337).
- `--repeats`: Anzahl Messläufe nach dem Warm-up (Standard: 3).
- `--symbol`: Symbol-Kennung für die Candle-Serie (Standard: `BTCUSDT`).

Das Skript führt keinen Datei- oder Netzwerk-I/O aus. Optional verfügbare
Metriken (`p95_stage_latency_ms`, `peak_rss_mb`) fallen auf `null`, wenn die
jeweiligen Abhängigkeiten nicht verfügbar sind.

## CI-Integration

Der Workflow `.github/workflows/ci.yml` enthält einen eigenen Job `perf`, der
nach erfolgreichem Testlauf ausgeführt wird:

1. `PYTHONPATH=src pytest -q -m slow tests/perf/test_budget_smoke.py`
   - stellt sicher, dass der Durchsatz mindestens 833 Kerzen pro Sekunde beträgt
     (≈ 50k/min) und die Stage-Latenz unter 50 ms bleibt, sofern gemessen.
2. `PYTHONPATH=src python scripts/bench_backtester.py --candles 50000 --repeats 2`
   - erzeugt `bench.json`, das als Artefakt `bench-report` hochgeladen wird.

Da der Job mit `continue-on-error: true` konfiguriert ist, blockiert er kein
Merge, dient jedoch als Frühwarnsystem bei Performance-Regressions.

## Artefakt auslesen

1. Öffne den `perf`-Job in GitHub Actions.
2. Lade das Artefakt **bench-report** herunter.
3. Analysiere die JSON-Datei lokal, z. B. mit `jq`:

```bash
jq '.' bench.json
```

Die Kennzahlen sollten gegen das definierte Budget verglichen werden. Signifikante
Abweichungen (z. B. `candles_per_sec.min < 900`) sollten eine Investigations-
Timeline auslösen.

## Lokale Schnelltests

Für eine schnelle Regression genügt der Smoke-Test mit reduzierten Candles:

```bash
WELTKLASSE_SKIP_PERF=0 PYTHONPATH=src pytest -q -m slow tests/perf/test_budget_smoke.py
```

Die Umgebungsvariable `WELTKLASSE_SKIP_PERF` wird in CI auf `1` gesetzt, um den
Smoke-Test im Haupt-Testjob auszuschalten; der separate `perf`-Job führt ihn
immer aus. Lokale Läufe können die Variable unset lassen, um die Performance-
Kontrolle aktiv zu halten.
