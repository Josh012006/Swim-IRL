#!/bin/bash
# .github/ci/train_phase0b_airl.sh
# Phase 0b -- AIRL training pipeline.
# 2x2 grid, same reduction/reasoning as train_phase0b_gcl.sh -- structure
# mirrors it exactly; differs only in the Python command invoked
# (phase0b_airl_training) and the service name (swim-irl-phase0b-airl).
# Run independently from GCL so a crash in one doesn't take the other
# down, and both can checkpoint/resume independently.

FLAG_FILE="${FLAG_FILE}"
SHA="${SHA}"
BRANCH="${BRANCH}"
WORK_DIR="${WORK_DIR}"
VENV="$WORK_DIR/.venv/bin"
N_ENVS="${N_ENVS:-2}"

cd "$WORK_DIR"

RUN_ID="$(date -u '+%Y%m%d_%H%M%S')_${SHA:0:7}"
export RUN_ID
log_prefix="[phase0b-airl:$RUN_ID]"

if [ -f ~/.sendgrid_config ]; then
  set -a
  source ~/.sendgrid_config
  set +a
fi
NOTIFY_EMAIL="josuesmjr.mongan@gmail.com"
FROM_EMAIL="josuesmjr.mongan@gmail.com"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $log_prefix $*"; }

send_email() {
  local subject="$1"
  local body="$2"
  if [ -z "$SENDGRID_API_KEY" ]; then
    log "SENDGRID_API_KEY not set — skipping email."
    return
  fi
  local escaped_body
  escaped_body=$(echo "$body" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
  curl -s -o /dev/null -X POST https://api.sendgrid.com/v3/mail/send \
    -H "Authorization: Bearer $SENDGRID_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"personalizations\":[{\"to\":[{\"email\":\"$NOTIFY_EMAIL\"}]}],\"from\":{\"email\":\"$FROM_EMAIL\"},\"subject\":\"$subject\",\"content\":[{\"type\":\"text/plain\",\"value\":\"$escaped_body\"}]}" || true
}

TRAIN_EASY_EASY=false
TRAIN_EASY_EASY_MEDIUM=false
TRAIN_MEDIUM_EASY=false
TRAIN_MEDIUM_EASY_MEDIUM=false

while IFS='=' read -r key value || [ -n "$key" ]; do
  [ -z "$key" ] && continue
  case "$key" in
    train_easy_easy)          TRAIN_EASY_EASY="$value" ;;
    train_easy_easy_medium)   TRAIN_EASY_EASY_MEDIUM="$value" ;;
    train_medium_easy)        TRAIN_MEDIUM_EASY="$value" ;;
    train_medium_easy_medium) TRAIN_MEDIUM_EASY_MEDIUM="$value" ;;
  esac
done < "$FLAG_FILE"

TRAINING_FAILED=false

log "Phase 0b AIRL training started. Run ID: $RUN_ID"
send_email "🚀 Swim-IRL Phase 0b AIRL training started ($SHA)" \
  "Phase 0b AIRL training started.\nRun ID: $RUN_ID\nBranch: $BRANCH\nCommit: $SHA"

cpu_watcher() {
  local low_count=0
  local threshold=20
  local max_low=10
  while true; do
    sleep 30
    local usage
    usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d',' -f1)
    usage=${usage%.*}
    if [ "${usage:-0}" -lt "$threshold" ] 2>/dev/null; then
      low_count=$((low_count + 1))
      if [ "$low_count" -ge "$max_low" ]; then
        send_email "⚠️ Swim-IRL Phase 0b AIRL — low CPU detected" \
          "CPU below ${threshold}% for 5 minutes.\nRun ID: $RUN_ID"
        low_count=0
      fi
    else
      low_count=0
    fi
  done
}

log_reporter() {
  local interval=$((4 * 3600))
  while true; do
    sleep $interval
    local body="Phase 0b AIRL progress — $(date -u '+%Y-%m-%d %H:%M UTC')\nRun ID: $RUN_ID\n\n"
    for cell in easy_easy easy_easy_medium medium_easy medium_easy_medium; do
      local logfile="$WORK_DIR/logs/phase0b_airl_${cell}.log"
      if [ -f "$logfile" ]; then
        body+="=== $cell ===\n$(tail -50 "$logfile")\n\n"
      fi
    done
    send_email "📊 Swim-IRL Phase 0b AIRL report — $(date -u '+%H:%M UTC')" "$body"
  done
}

cpu_watcher &
CPU_WATCHER_PID=$!
log_reporter &
LOG_REPORTER_PID=$!

cleanup() {
  kill "$CPU_WATCHER_PID" 2>/dev/null || true
  kill "$LOG_REPORTER_PID" 2>/dev/null || true
}
trap cleanup EXIT

run_cell() {
  local model="$1"
  local seed_mode="$2"
  local cell="${model}_${seed_mode}"
  local logfile="$WORK_DIR/logs/phase0b_airl_${cell}.log"

  log "Starting AIRL cell $cell..."
  send_email "🟢 Swim-IRL AIRL — ${cell} started" \
    "AIRL cell ${cell} started.\nRun ID: $RUN_ID\nCommit: $SHA"

  if RUN_ID="$RUN_ID" "$VENV/python" -u -m experiments.phase0b_airl_training \
      --seed 0 --n-envs "$N_ENVS" --cell "${model}_${seed_mode}" \
      > "$logfile" 2>&1; then
    touch "$WORK_DIR/logs/phase0b_airl_${cell}.DONE"
    send_email "✅ Swim-IRL AIRL — ${cell} complete" \
      "AIRL cell ${cell} finished.\nRun ID: $RUN_ID"
    log "AIRL cell $cell complete."
  else
    touch "$WORK_DIR/logs/phase0b_airl_${cell}.FAILED"
    send_email "❌ Swim-IRL AIRL — ${cell} FAILED" \
      "AIRL cell ${cell} failed.\nRun ID: $RUN_ID\n\nLast logs:\n$(tail -50 "$logfile")"
    log "AIRL cell $cell FAILED."
    TRAINING_FAILED=true
  fi
}

mkdir -p "$WORK_DIR/logs"

[ "$TRAIN_EASY_EASY"          = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell easy easy
[ "$TRAIN_EASY_EASY_MEDIUM"   = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell easy easy_medium
[ "$TRAIN_MEDIUM_EASY"        = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell medium easy
[ "$TRAIN_MEDIUM_EASY_MEDIUM" = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell medium easy_medium

log "Resetting train_phase0b_airl.flag..."
{
  echo "train=false"
  echo "train_easy_easy=false"
  echo "train_easy_easy_medium=false"
  echo "train_medium_easy=false"
  echo "train_medium_easy_medium=false"
} > "$FLAG_FILE"

log "Committing results..."
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add experiments/results/ .github/ci/train_phase0b_airl.flag
if [ -z "$(git status --porcelain)" ]; then
  log "No changes to commit."
else
  git commit -m "feat: Phase 0b AIRL results after commit $SHA [skip ci]"
  git push origin "$BRANCH"
  log "Pushed to branch $BRANCH."
fi

log "Creating GitHub issue..."
BODY="Phase 0b AIRL training from commit: $SHA"$'\n\n'
for cell in easy_easy easy_easy_medium medium_easy medium_easy_medium; do
  if [ -f "$WORK_DIR/logs/phase0b_airl_${cell}.DONE" ]; then
    BODY+="✅ ${cell} complete"$'\n'
  elif [ -f "$WORK_DIR/logs/phase0b_airl_${cell}.FAILED" ]; then
    BODY+="❌ ${cell} FAILED"$'\n'
  fi
done
BODY+=$'\n''- [ ] Review TensorBoard logs'$'\n'
BODY+='- [ ] Compare AIRL results to GCL'$'\n'
BODY+='- [ ] Update README with Phase 0b results'

gh issue create \
  --title "Phase 0b AIRL training finished ($SHA)" \
  --body "$BODY"

send_email "🏁 Swim-IRL Phase 0b AIRL — all cells done" \
  "All requested AIRL cells completed.\n\n$BODY\n\nRun ID: $RUN_ID\nBranch: $BRANCH"
log "All done."