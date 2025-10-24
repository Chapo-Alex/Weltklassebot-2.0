# CI Runbook

## Job-Übersicht

Der Workflow [`CI`](../.github/workflows/ci.yml) besteht aus fünf Jobs, die
sequentiell bzw. optional ausgeführt werden:

1. **Ruff** – Linting via `ruff check` nach Installation der Dev-Abhängigkeiten.
2. **Mypy** – Typprüfung mit `mypy --strict src` (abhängig von Ruff).
3. **Pytest** – Führt `python scripts/run_tests_cov.py --junitxml=...` aus, erzeugt
   `coverage.xml` und lädt Coverage-, JUnit- und Metrik-Artefakte hoch. Die
   Umgebungsvariablen `PYTHONHASHSEED=0` und `WELTKLASSE_SKIP_PERF=1` sorgen für
   deterministische Hashes und überspringen Performance-Tests.
4. **Performance** – Optionaler Job (`continue-on-error: true`), der den Slow-Test
   `tests/perf/test_budget_smoke.py` sowie den Benchmark-Skriptlauf startet und
   das Artefakt `bench-report` bereitstellt.
5. **Mutation** – Ebenfalls optional (`continue-on-error: true`). Führt Cosmic Ray
   mit der Konfiguration `ops/mutation/cosmic-ray.toml` aus und speichert den
   Report `mutation-report.txt`.

## Coverage Gate

Der Coverage-Gate-Schritt läuft nur im Pytest-Job der CI. Er liest `coverage.xml`
und vergleicht Gesamt- und Paket-Coverage (`core`, `portfolio`, `risk`,
`strategy`) gegen den Schwellwert, der in `pyproject.toml` unter
`[tool.reichmacher] coverage_threshold` konfiguriert ist (derzeit 0.9 → 90 %).

### Lokale Ausführung ohne Gate

Lokale Developer-Runs können das Gate auslassen, indem nur das Testskript
aufgerufen wird:

```bash
PYTHONPATH=src python scripts/run_tests_cov.py
```

Der Coverage-Bericht wird dennoch erzeugt (`coverage.xml`), aber kein Threshold-
Check ausgeführt. Alternativ können gezielte Teil-Suites direkt mit `pytest -q`
oder Markern gestartet werden, wenn kein Coverage benötigt wird.

## Selektive Läufe

- **Linting**: `ruff check src tests`.
- **Typing**: `mypy --strict src`.
- **Mutation lokal** (langsam):
  ```bash
  PYTHONPATH=src cosmic-ray run ops/mutation/cosmic-ray.toml cr.sqlite
  cosmic-ray report cr.sqlite | less
  ```
- **Performance**: siehe [Performance Runbook](runbook-perf.md).

## Troubleshooting

- **Fail wegen Coverage**: Überprüfe `coverage.xml` (z. B. mit `coverage report`)
  und fokussiere fehlende Module. Die CI-Ausgabe listet Pakete unterhalb des
  Schwellwerts.
- **Mutation-Job rot**: Da `continue-on-error` aktiv ist, blockiert dies keinen
  Merge. Lade das Artefakt `mutation-report` zur Analyse.
- **Optionales NumPy**: Tests mit Marker `needs_numpy` werden automatisch
  übersprungen, wenn NumPy nicht verfügbar ist. Dies beeinflusst den Coverage-Wert
  nicht, solange genügend andere Tests aktiv bleiben.
