#!/bin/bash
# Keep the live-forward lane accruing.
#
# Refreshes the last few days of load and weather, and archives the raw EIA
# responses as vintages. Safe to run hourly: every write is idempotent, and
# re-fetching is exactly how EIA's in-place revisions get captured.
#
# This deliberately does NOT issue forecasts. Issuance belongs to the Modal
# deployment, which owns the durable ledger volume; running a second bake path
# from a laptop would create two ways to write the same run.
#
# This is a stopgap for one machine. A gap in this schedule is a permanent gap
# in the record, because a vintage nobody captured cannot be reconstructed.
#
# To run it hourly on macOS, install a LaunchAgent that calls this script with
# StartInterval 3600. macOS blocks LaunchAgents from reading external volumes
# by default: the job exits 126 with "Operation not permitted" while the same
# command works from a terminal. Grant Full Disk Access to /bin/bash in
# System Settings > Privacy & Security, or keep the checkout on the internal
# disk. Verify with: launchctl list | grep surge  (the middle column is the
# last exit status, and it must be 0).
set -euo pipefail

REPO="${SURGE_REPO:-/Volumes/Extreme Pro/surge}"
PYTHON="${SURGE_PYTHON:-$REPO/.venv/bin/python}"
LOG_DIR="${SURGE_LOG_DIR:-$HOME/.surge/logs}"
RTOS="PJM CISO ERCO MISO NYIS ISNE SWPP"

cd "$REPO"
mkdir -p "$LOG_DIR"

if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO/.env"
  set +a
fi

export SURGE_DATA_DIR="${SURGE_DATA_DIR:-$HOME/.surge/data}"
export SURGE_CODE_REVISION="${SURGE_CODE_REVISION:-$(git rev-parse HEAD)}"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(stamp)] ingest start" >> "$LOG_DIR/ingest.log"
# shellcheck disable=SC2086
"$PYTHON" -m surge.ingest --bas $RTOS --days 4 >> "$LOG_DIR/ingest.log" 2>&1
"$PYTHON" - <<'PYEOF' >> "$LOG_DIR/ingest.log" 2>&1
from surge import vintage
print(f"vintage archive entries: {len(vintage.read_index())}")
PYEOF
echo "[$(stamp)] ingest ok" >> "$LOG_DIR/ingest.log"
