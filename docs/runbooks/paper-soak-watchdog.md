# Paper-soak watchdog runbook

**Purpose:** keep the 24/7 paper-trading loop alive and surface fail-closed
events to an operator within ~5 minutes.

This runbook wires three pieces that already exist on the platform:

1. `runtime supervise` — the long-running cycle driver.
2. `scripts/extract_status.py` — JSON heartbeat reader.
3. `scripts/alert_failures.py` — structured-log scanner that exits non-zero
   on fresh failures (see the script for the supported event suffixes).

The goal is *operator visibility*, not pager-grade SLOs.

---

## 1. Foreground: keep `supervise` running

The supervise loop is the platform's normal long-runner. It already
fail-closes on broker disconnects and stale data via
`engines/session/cycle_guards.py`; the watchdog only needs to keep the
process alive and notify on restart.

### Linux / WSL (systemd)

Drop into `/etc/systemd/system/quant-paper.service`:

```ini
[Unit]
Description=Quant paper-trading supervise loop
After=network-online.target docker.service
Wants=network-online.target

[Service]
WorkingDirectory=/opt/quant
EnvironmentFile=/opt/quant/.env
ExecStart=/opt/quant/.venv/bin/python -m quant_platform supervise \
  --mode paper \
  --execution-backend ib-paper \
  --contracts-file /opt/quant/infra/config/universe_300.json \
  --interval 300
Restart=always
RestartSec=15
StandardOutput=append:/var/log/quant/paper.log
StandardError=append:/var/log/quant/paper.log

[Install]
WantedBy=multi-user.target
```

Enable: `systemctl enable --now quant-paper`. Confirm with `systemctl status
quant-paper` and `journalctl -u quant-paper -f`.

### Windows Task Scheduler

Until the operator host moves off Windows, run inside a PowerShell wrapper
that restarts on exit:

```powershell
# scripts/run_supervise_windows.ps1
while ($true) {
  & "$PSScriptRoot\..\.venv\Scripts\python.exe" -m quant_platform supervise `
    --mode paper --execution-backend ib-paper `
    --contracts-file "$PSScriptRoot\..\infra\config\universe_300.json" `
    --interval 300
  Start-Sleep -Seconds 15
}
```

Register as a "Run on startup" task with the user account that owns the
project directory.

---

## 2. Heartbeat: detect a wedged loop

`runtime supervise` writes
`data/parquet/_status/supervise_status.json` every cycle (atomic .tmp +
rename, mirrors the pattern from `scripts/extract_status.py`). The status
JSON carries `last_cycle_at_utc`, `cycle_count`, and the last `event`.

### Cron, every 5 minutes

```cron
*/5 * * * * /opt/quant/.venv/bin/python /opt/quant/scripts/extract_status.py \
    --path /opt/quant/data/parquet/_status/supervise_status.json \
    --stuck-after-minutes 15 \
    || curl -sf -X POST -H "Content-Type: text/plain" \
       --data-binary @- https://your-webhook/alerts
```

Exit-code contract (preserved from existing usage):

| Exit | Meaning                                      |
|------|----------------------------------------------|
| 0    | status fresh                                  |
| 1    | status missing OR older than `--stuck-after-minutes` |
| 2    | malformed status JSON                         |

---

## 3. Fail-closed events: scan the log

Wrap `scripts/alert_failures.py` in the same cron with a 5-minute window:

```cron
*/5 * * * * /opt/quant/.venv/bin/python /opt/quant/scripts/alert_failures.py \
    --since-minutes 5 /var/log/quant/ \
    || /opt/quant/scripts/notify-operator.sh
```

`alert_failures.py` follows the platform structured-event convention —
event suffixes `.fail_closed`, `.halted`, `.kill_switch` (CRITICAL) and
`.failure`, `.error`, `.disconnected`, `.rejected` (ERROR). When the
catalog grows, extend the suffix lists at the top of the script rather
than adding parallel scanners.

### Manual triage

```bash
# Last hour, human-readable digest:
python scripts/alert_failures.py --since-minutes 60 data/logs/

# Machine-readable feed for piping into jq:
python scripts/alert_failures.py --json data/logs/ | jq '.incidents[] | select(.severity=="CRITICAL")'
```

---

## 4. Channel: where alerts land

For the single-VPS paper setup, a stdout webhook is enough. Suggested
order of escalation:

1. **Local file** — `tee -a /var/log/quant/alerts.log` for retention.
2. **Discord/Slack webhook** — operator-visible (`notify-operator.sh`
   wraps `curl -X POST`).
3. **Email-as-last-resort** — only when the webhook itself goes down
   (configure a dead-man's switch via Healthchecks.io against the cron
   running `extract_status.py`).

No PagerDuty integration is wired today — the platform is paper-only
and `production_candidate` gating already blocks live trading on
failed assertions, so the urgency bar is "operator looks within an
hour", not "wake someone up".

---

## 5. Verification — quarterly drill

Once a quarter, run through this checklist:

- [ ] `systemctl stop quant-paper` → within 15 min the heartbeat cron
      flips to exit 1 and the webhook fires.
- [ ] Synthesize a `text_extractor.failure` line into a fresh log file
      → `alert_failures.py --since-minutes 5` exits 1 and prints it.
- [ ] Restart the service → heartbeat returns to exit 0 within one
      cycle (5 min).
- [ ] Confirm the previous alert event was acknowledged (manual — the
      script is stateless, the *channel* owns dedupe).
