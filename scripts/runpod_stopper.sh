#!/usr/bin/env bash
# On-pod hard stop (P-69, 2026-10-06): session E's ad-hoc /workspace/stopper.sh made permanent (traps 51 addendum), so a pod's
# bill is capped on the pod itself -- the chain's runtime guard defaults to 40 h, its ship loop can sleep, and laptop watchers
# die when the laptop sleeps (traps 51 / 53).
#
# Usage (on the pod, right after creation and BEFORE the chain; the deadline is absolute, in epoch seconds):
#   setsid nohup bash scripts/runpod_stopper.sh /workspace/job_v15c.log $(( POD_T0 + 12060 )) \
#     > /workspace/stopper.log 2>&1 < /dev/null &
# It calls the GraphQL podStop when
#   (a) the job log shows the chain's last line ("job done:", or the AUTO_STOP trap's "job exit rc="), 90 s later, as a
#       backup to the chain's own self-stop;
#   (b) the clock passes the deadline (the money cap: hours x $0.74);
#   (c) the file /workspace/STOP_NOW exists (a manual stop that needs no RunPod API call from the laptop).
# The pod's own key and id come from PID 1's environment, never the ssh session's (traps 51).
set -u
LOG="${1:?usage: runpod_stopper.sh <job_log> <deadline_epoch_s>}"
DEADLINE="${2:?usage: runpod_stopper.sh <job_log> <deadline_epoch_s>}"
POLL="${STOPPER_POLL_S:-30}"

eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
  log "!! RUNPOD_API_KEY / RUNPOD_POD_ID not in /proc/1/environ -- this stopper cannot stop the pod"
  exit 1
fi

stop_pod() {
  log "podStop: $1"
  local t out
  for t in 1 2 3 4 5 6; do
    out=$(curl -s -m 30 -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
          -d "{\"query\":\"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}")
    log "  try $t: $out"
    echo "$out" | grep -q '"desiredStatus"' && exit 0
    sleep 20
  done
  log "!! podStop failed 6 times -- stop the pod from the laptop (RunPod MCP stop-pod)"
  exit 2
}

log "stopper armed: pod ${RUNPOD_POD_ID}, deadline $(date -u -d "@${DEADLINE}" '+%Y-%m-%d %H:%M:%S') UTC" \
    "($(( (DEADLINE - $(date +%s)) / 60 )) min from now), job log ${LOG}, poll ${POLL} s"
while :; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && stop_pod "deadline reached"
  [ -f /workspace/STOP_NOW ] && stop_pod "/workspace/STOP_NOW exists"
  if [ -f "$LOG" ] && grep -qE "job done:|job exit rc=" "$LOG"; then
    sleep 90
    stop_pod "the job finished ($(grep -E 'job done:|job exit rc=' "$LOG" | tail -1))"
  fi
  sleep "$POLL"
done
