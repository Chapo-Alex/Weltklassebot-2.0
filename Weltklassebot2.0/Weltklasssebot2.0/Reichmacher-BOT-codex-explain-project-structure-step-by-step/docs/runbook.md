# Weltklassebot Production Runbook

This runbook captures the operational procedures for starting, stopping, and
troubleshooting Weltklassebot deployments. Pair it with the alert rules under
`ops/alerts/` and the CI status checks to maintain deterministic, observable
roll-outs.

## Start and Stop Procedures

1. Verify that the target host has the latest artefacts and data snapshots
   replicated locally.
2. Export the required environment (see [Environment Configuration](#environment-configuration)).
3. Start the service:
   ```bash
   systemctl start weltklassebot
   ```
4. Confirm liveness and metrics availability via:
   ```bash
   systemctl status weltklassebot
   curl -sf http://localhost:${WELTKLASSE_METRICS_PORT:-9000}/metrics | head
   ```
5. To stop the service gracefully:
   ```bash
   systemctl stop weltklassebot
   ```
6. For a rolling restart during mitigations:
   ```bash
   systemctl restart weltklassebot
   ```

Escalate to the on-call lead if the service fails to start twice in succession
or emits non-deterministic outputs after restart.

## Environment Configuration

Set the following environment variables before launching the service:

- `WELTKLASSE_METRICS_PORT`: Prometheus exporter port (default `9000`).
- `HTTP_PROXY` / `HTTPS_PROXY`: Proxy endpoints required for outbound package
  resolution in restricted environments.
- `PIP_INDEX_URL` / `PIP_TRUSTED_HOST`: Internal PyPI mirror configuration to
  ensure deterministic builds.
- `PYTHONHASHSEED=0`: Guarantees stable hash randomisation across restarts.

Exporting these variables in the systemd unit or deployment manifest keeps the
runtime reproducible and the metrics endpoint reachable.

## Seeds and Reproducibility

All backtests and live simulations accept a deterministic seed via the unified
CLI:

```bash
python -m cli backtest --seed 123 --data-snapshot /var/lib/weltklassebot/snapshots/latest.parquet
```

For forensic investigations, capture the seed used during the incident along
with the CLI invocation, config file, and strategy version. Re-running the same
command with identical inputs must reproduce the fills and equity curve exactly.
If reproducibility fails, open a `determinism` severity incident and page the
Quant/Infra rotation.

## Data Snapshots

Data snapshots are stored under `/var/lib/weltklassebot/snapshots/` with ISO8601
prefixes. Each deployment must reference a consistent snapshot across all
replicas to avoid skew in order sequencing. To refresh snapshots:

1. Download the new dataset to a staging directory.
2. Validate timestamps with `python -m cli backtest --data-snapshot <path> --seed 42`.
3. Atomically update the symlink `snapshots/latest.parquet` to the new file.
4. Restart the service during a maintenance window.

Retain at least three previous snapshots for rollback scenarios.

## Service Level Objectives

- **Latency SLO**: p95 `stage_latency_seconds` < 0.25 s sustained per 5-minute
  window.
- **Idempotency SLO**: cumulative duplicate fills remain at 0. Any non-zero value
  triggers an on-call investigation.
- **Determinism SLO**: repeated backtests with identical seeds must produce the
  same equity hash (`tests/determinism/test_equity_hash.py` parity).

Breaching any SLO requires raising a reliability incident with impact analysis
and remediation timeline.

## Alert Triage Playbooks

### HighStageLatency (severity=page)

- **Trigger**: `stage_latency_seconds` p95 exceeds 250 ms for 5 minutes.
- **Immediate actions**:
  1. Inspect recent deploys or config changes for increased load.
  2. Check system metrics (CPU, memory, IO wait) on affected hosts.
  3. Compare against baseline snapshots; if latency persists, fail over to a
     warm standby or scale out replicas.
- **Follow-up**: file a performance regression ticket with profile traces.

### RiskCooldownTooLong (severity=warn)

- **Trigger**: `risk_state{state="COOL"}` remains at 1 for 30 minutes.
- **Immediate actions**:
  1. Confirm trading hours and scheduled maintenance windows.
  2. Review recent risk denials for chronic exposure spikes.
  3. Ensure the risk store is rotating (`state/jsonl_store.py`) and that recent
     writes are succeeding (check disk space and permissions).
- **Follow-up**: adjust thresholds if intended behaviour or initiate risk review
  if unexpected.

### DrawdownNearLimit (severity=warn)

- **Trigger**: Drawdown exceeds 90 % of the configured cap for 10 minutes.
- **Immediate actions**:
  1. Alert the strategy owner and confirm portfolio exposure.
  2. Validate that fee/slippage models match production venue settings.
  3. Prepare contingency actions (reduce positions, halt trading) if losses
     accelerate.
- **Follow-up**: document mitigation results and update the runbook with any
  novel remediation steps.

## Incident Response Checklist

1. Acknowledge alerts in the monitoring system.
2. Capture logs, metrics snapshots, and the active seed/config bundle.
3. Initiate a deterministic replay using the captured seed and snapshot.
4. If reproduction diverges, escalate to Engineering leads and halt trading.
5. After mitigation, document findings in this runbook and schedule a postmortem.
