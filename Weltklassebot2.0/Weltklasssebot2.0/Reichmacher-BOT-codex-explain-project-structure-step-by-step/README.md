# Weltklassebot

Weltklassebot is a deterministic research harness for breakout-biased crypto strategies.
It contains a reproducible backtesting engine, portfolio accounting primitives, and a
risk manager that can be embedded into higher-level trading infrastructure.

## Install & Test

```bash
pip install -e .[dev]
pytest
```

## CLI

The Typer-based entrypoint provides reproducible backtesting, walk-forward, and tuning
workflows from a single command surface. Global flags such as `--seed`,
`--data-snapshot`, `--prom-port`, and `--config` may be supplied before or after the
sub-command to control deterministic behaviour and shared defaults.

```bash
python -m cli backtest --seed 123 --data data/btc.parquet \
  --start 2024-01-01T00:00:00Z --end 2024-01-07T00:00:00Z

python -m cli walkforward --config configs/walkforward.toml

python -m cli tune --config configs/tune.toml --params configs/params.json
```

When no dataset is provided the `backtest` command falls back to the deterministic
synthetic harness used by regression tests, making `python -m cli backtest --seed 123`
a quick smoke check. Legacy scripts under `scripts/` now emit a deprecation notice and
delegate to the unified CLI.

## Reproduzierbare Runs

Wiederholbare Ergebnisse hängen von konsistenten Seeds, Daten-Snapshots und
Umgebungsvariablen ab. Verwende den vereinheitlichten CLI-Einstiegspunkt, um
einen Lauf vollständig zu fixieren:

```bash
python -m cli backtest \
  --seed 123 \
  --data-snapshot /var/lib/weltklassebot/snapshots/latest.parquet \
  --prom-port ${WELTKLASSE_METRICS_PORT:-9000}
```

Archiviert zusätzlich die verwendete Konfigurationsdatei (`--config`) und den
Commit-Hash, damit Simulationen und Produktionsreplays deckungsgleich bleiben.

## Operations

Ausführliche Betriebs- und Incident-Prozesse findest du im
[Runbook](docs/runbook.md). Es enthält Start/Stop-Anleitungen, definierte SLOs
und Alarm-Triage-Playbooks für den Regelbetrieb.

### Hinter Proxy / Mirror bauen

```bash
export HTTP_PROXY=http://user:pass@proxy.corp:8080
export HTTPS_PROXY=http://user:pass@proxy.corp:8080
export PIP_INDEX_URL=https://artifactory.corp/pypi/simple
export PIP_TRUSTED_HOST=artifactory.corp
pip install -e .[dev]
```

### Ohne Build-Isolation

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev] --no-build-isolation
```

## Observability

A lightweight Prometheus exporter is provided to surface runtime metrics and drive
alerting automation.

### Metrics exporter

Run the exporter alongside the trading stack to expose metrics on `:9000`:

```bash
python scripts/exporter.py
```

The exporter shares the global registry used by the engine, portfolio, and risk
manager. Metrics include stage latencies, realised PnL, drawdown, fill flow, and
risk-gate state transitions.

### Prometheus scrape configuration

Configure Prometheus to scrape the exporter and apply the bundled alerting rules:

```yaml
scrape_configs:
  - job_name: "weltklassebot"
    static_configs:
      - targets: ["localhost:9000"]
    metrics_path: /metrics
rule_files:
  - ops/alerts/weltklasse-rules.yaml
```

### Alert runbook

The PrometheusRule `ops/alerts/weltklasse-rules.yaml` defines three actionable alerts:

- **HighStageLatency** (`severity=page`): p95 latency for any engine stage exceeds
  250 ms for five minutes.
  - Investigate recent releases for performance regressions.
  - Inspect strategy logs for pathologically slow market data processing.
  - Consider diverting flow to a standby engine if remediation requires a restart.

- **RiskCooldownTooLong** (`severity=warn`): the risk state remains in `COOLDOWN`
  for 30 minutes.
  - Confirm that the trade-per-day threshold and cooldown parameters are correct.
  - Verify that the risk manager is still receiving portfolio snapshots.
  - If intentional (e.g., scheduled maintenance), acknowledge the alert.

- **DrawdownNearLimit** (`severity=warn`): observed drawdown exceeds 90 % of the
  configured maximum for ten minutes.
  - Validate the `cfg_max_dd` metric against the deployment configuration.
  - Engage the strategy owner to assess whether positions should be reduced.
  - If drawdown continues to deteriorate, prepare for a risk-induced halt.

For operational emergencies, restart the trading stack with:

```bash
systemctl restart weltklassebot
```

To restore state after a failure, load the most recent deterministic data snapshot
from cold storage and replay fills through the backtester to rebuild the portfolio
state. Document any mitigation steps in the runbook so on-call engineers can
quickly iterate on fixes.

## Mutation testing

A focused Cosmic Ray configuration lives under `ops/mutation/cosmic-ray.toml` and
targets the core execution packages. The CI workflow executes the suite nightly
on a best-effort basis and uploads `mutation-report.txt` as a build artefact.

To run the same check locally (it can take several minutes), execute:

```bash
PYTHONPATH=src cosmic-ray run ops/mutation/cosmic-ray.toml cr.sqlite
cosmic-ray report cr.sqlite
```

You can restrict scope by editing the `modules` entry in the configuration file
to speed up local investigations.

## Development

Install the development extras in editable mode and enable the pre-commit hooks:

```bash
pip install -e ".[dev]"
pre-commit install
```

Frequently used project targets are available via the `Makefile`:

```bash
make test      # run the fast unit and property tests (excludes slow markers)
make cov       # execute the suite with coverage collection
make bench     # launch the deterministic backtester benchmark
```

## Devcontainer & Docker

The repository ships with a development container based on `docker/Dockerfile`.
Launch VS Code with the included `.devcontainer/devcontainer.json` configuration
to start an environment that installs the development extras automatically. To
exercise the same setup manually, build and run the image:

```bash
docker build -t weltklassebot-dev -f docker/Dockerfile .
docker run --rm -it weltklassebot-dev
```

Für proxied oder gemirrorte Netzwerke und Builds ohne Isolation stehen optionale
Build-Argumente zur Verfügung:

```bash
docker build \
  --build-arg HTTP_PROXY=http://user:pass@proxy.corp:8080 \
  --build-arg HTTPS_PROXY=http://user:pass@proxy.corp:8080 \
  --build-arg PIP_INDEX_URL=https://artifactory.corp/pypi/simple \
  --build-arg PIP_TRUSTED_HOST=artifactory.corp \
  -t weltklassebot:dev .
```

```bash
docker build --build-arg SWITCH_NO_ISOLATION=true -t weltklassebot:dev .
```

The container entrypoint executes the deterministic test suite using
`PYTHONHASHSEED=0`, mirroring the behaviour in CI.
