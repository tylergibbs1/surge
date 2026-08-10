#!/usr/bin/env bash
# Sync the working tree to the GPU pod and score one experiment.
#
#   bash scripts/run_remote_experiment.sh <exp_name> '<config_json>'
#
# Prints the runner's METRIC: line on stdout.
#
# The scoring harness and its leakage referee are deliberately NOT synced:
# an autonomous search that can edit its own grader can pass by editing the
# grader. The pod keeps pristine copies of causal_guard.py and
# research_eval.py, so every candidate is judged by the same referee.
set -uo pipefail

POD_HOST="${SURGE_POD_HOST:-root@213.173.103.224}"
POD_PORT="${SURGE_POD_PORT:-39713}"
POD_KEY="${SURGE_POD_KEY:-$HOME/.ssh/id_ed25519}"
POD_REPO="${SURGE_POD_REPO:-/workspace/surge}"
POD_DATA="${SURGE_POD_DATA:-/root/data}"

EXP_NAME="${1:?usage: run_remote_experiment.sh <exp_name> <config_json>}"
CONFIG_JSON="${2:-{\}}"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$POD_PORT" -i "$POD_KEY")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -rlptz -e "ssh ${SSH_OPTS[*]}" \
  --exclude '__pycache__' \
  --exclude 'causal_guard.py' \
  --exclude 'research_eval.py' \
  "$REPO_ROOT/experiments/" "$POD_HOST:$POD_REPO/experiments/" >/dev/null 2>&1

rsync -rlptz -e "ssh ${SSH_OPTS[*]}" \
  --exclude '__pycache__' \
  "$REPO_ROOT/src/" "$POD_HOST:$POD_REPO/src/" >/dev/null 2>&1

ssh "${SSH_OPTS[@]}" "$POD_HOST" \
  "cd $POD_REPO && SURGE_DATA_DIR=$POD_DATA PYTHONPATH=$POD_REPO PYTHONUNBUFFERED=1 \
   python3 -m experiments.research_eval $(printf '%q' "$EXP_NAME") $(printf '%q' "$CONFIG_JSON")"
rc=$?

# A crash must not read as "no improvement" — surface it as an explicit failure.
if [ $rc -ne 0 ]; then
  echo "METRIC: {\"exp\": \"$EXP_NAME\", \"status\": \"error\", \"exit_code\": $rc}"
fi
exit $rc
