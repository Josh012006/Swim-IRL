#!/bin/bash
# .github/ci/train_phase0b_gcl.sh
# Phase 0b -- Guided Cost Learning training pipeline.
# Runs as a systemd service on the remote machine, completely
# independent of the GitHub Actions job lifecycle. Environment
# variables are loaded by systemd from /tmp/swim-irl-phase0b-gcl.env.
#
# 2x2 grid (easy/medium agent x easy/easy_medium seed_mode) -- "hard"
# was dropped from both axes after hard training completed but did not
# converge to an optimal policy. See README Limitations.
#
# Structure mirrors NanoGoal-RL's .github/ci/train.sh exactly:
# - Unique RUN_ID per invocation
# - Optional SendGrid email notifications at key milestones
# - CPU watcher (alerts if training silently crashes)
# - 4h log reporter
# - Per-cell sequential training, evaluation, plot generation
# - Auto-commit of results back to the branch
# - GitHub issue created at completion

FLAG_FILE="${FLAG_FILE}"
SHA="${SHA}"
BRANCH="${BRANCH}"
WORK_DIR="${WORK_DIR}"
VENV="$WORK_DIR/.venv/bin"
NANOGOAL_PATH="$WORK_DIR/external/NanoGoal-RL"
N_ENVS="${N_ENVS:-2}"

cd "$WORK_DIR"

RUN_ID="$(date -u '+%Y%m%d_%H%M%S')_${SHA:0:7}"
export RUN_ID
log_prefix="[phase0b-gcl:$RUN_ID]"

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

while IFS='=' read -r key value; do
  [ -z "$key" ] && continue
  case "$key" in
    train_easy_easy)          TRAIN_EASY_EASY="$value" ;;
    train_easy_easy_medium)   TRAIN_EASY_EASY_MEDIUM="$value" ;;
    train_medium_easy)        TRAIN_MEDIUM_EASY="$value" ;;
    train_medium_easy_medium) TRAIN_MEDIUM_EASY_MEDIUM="$value" ;;
  esac
done < "$FLAG_FILE"

TRAINING_FAILED=false

log "Phase 0b GCL training started. Run ID: $RUN_ID"
log "Flags:"
log "  easy/easy=$TRAIN_EASY_EASY  easy/easy_medium=$TRAIN_EASY_EASY_MEDIUM"
log "  medium/easy=$TRAIN_MEDIUM_EASY  medium/easy_medium=$TRAIN_MEDIUM_EASY_MEDIUM"
log "  n_envs=$N_ENVS"

send_email "🚀 Swim-IRL Phase 0b GCL training started ($SHA)" \
  "Phase 0b GCL training started.\nRun ID: $RUN_ID\nBranch: $BRANCH\nCommit: $SHA\n\nCells:\n- easy/easy=$TRAIN_EASY_EASY\n- easy/easy_medium=$TRAIN_EASY_EASY_MEDIUM\n- medium/easy=$TRAIN_MEDIUM_EASY\n- medium/easy_medium=$TRAIN_MEDIUM_EASY_MEDIUM"

cpu_watcher() {
  local low_count=0
  local threshold=20
  local max_low=10
  log "[CPU watcher] Started."
  while true; do
    sleep 30
    local usage
    usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d',' -f1)
    usage=${usage%.*}
    if [ "${usage:-0}" -lt "$threshold" ] 2>/dev/null; then
      low_count=$((low_count + 1))
      log "[CPU watcher] Low CPU: ${usage}% (${low_count}/${max_low})"
      if [ "$low_count" -ge "$max_low" ]; then
        log "[CPU watcher] ⚠️ CPU has been low for 5 minutes."
        send_email "⚠️ Swim-IRL Phase 0b GCL — low CPU detected" \
          "CPU below ${threshold}% for 5 consecutive minutes.\nRun ID: $RUN_ID\nCheck: tail -f $WORK_DIR/logs/phase0b_gcl_session.log"
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
    local body="Phase 0b GCL progress — $(date -u '+%Y-%m-%d %H:%M UTC')\nRun ID: $RUN_ID\n\n"
    for cell in easy_easy easy_easy_medium medium_easy medium_easy_medium; do
      local logfile="$WORK_DIR/logs/phase0b_gcl_${cell}.log"
      if [ -f "$logfile" ]; then
        body+="=== $cell ===\n$(tail -50 "$logfile")\n\n"
      fi
    done
    send_email "📊 Swim-IRL Phase 0b GCL report — $(date -u '+%H:%M UTC')" "$body"
    log "[Log reporter] 4h report sent."
  done
}

cpu_watcher &
CPU_WATCHER_PID=$!
log_reporter &
LOG_REPORTER_PID=$!

cleanup() {
  log "Stopping background processes..."
  kill "$CPU_WATCHER_PID" 2>/dev/null || true
  kill "$LOG_REPORTER_PID" 2>/dev/null || true
}
trap cleanup EXIT

run_cell() {
  local model="$1"
  local seed_mode="$2"
  local cell="${model}_${seed_mode}"
  local logfile="$WORK_DIR/logs/phase0b_gcl_${cell}.log"

  log "Starting GCL cell $cell..."
  send_email "🟢 Swim-IRL GCL — ${cell} started" \
    "GCL cell ${cell} started.\nRun ID: $RUN_ID\nCommit: $SHA"

  if RUN_ID="$RUN_ID" "$VENV/python" -u -m experiments.phase0b_gcl_training \
      --seed 0 --n-envs "$N_ENVS" --cell "${model}_${seed_mode}" \
      > "$logfile" 2>&1; then

    touch "$WORK_DIR/logs/phase0b_gcl_${cell}.DONE"
    send_email "✅ Swim-IRL GCL — ${cell} complete" \
      "GCL cell ${cell} finished.\nRun ID: $RUN_ID\nCommit: $SHA"
    log "GCL cell $cell complete."
  else
    touch "$WORK_DIR/logs/phase0b_gcl_${cell}.FAILED"
    send_email "❌ Swim-IRL GCL — ${cell} FAILED" \
      "GCL cell ${cell} failed.\nRun ID: $RUN_ID\nCommit: $SHA\n\nLast logs:\n$(tail -50 "$logfile")"
    log "GCL cell $cell FAILED."
    TRAINING_FAILED=true
  fi
}

mkdir -p "$WORK_DIR/logs"

[ "$TRAIN_EASY_EASY"          = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell easy easy
[ "$TRAIN_EASY_EASY_MEDIUM"   = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell easy easy_medium
[ "$TRAIN_MEDIUM_EASY"        = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell medium easy
[ "$TRAIN_MEDIUM_EASY_MEDIUM" = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell medium easy_medium

log "Resetting train_phase0b_gcl.flag..."
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
git add experiments/results/ .github/ci/train_phase0b_gcl.flag
if [ -z "$(git status --porcelain)" ]; then
  log "No changes to commit."
else
  git commit -m "feat: Phase 0b GCL results after commit $SHA [skip ci]"
  git push origin "$BRANCH"
  log "Pushed to branch $BRANCH."
fi

log "Creating GitHub issue..."
BODY="Phase 0b GCL training from commit: $SHA"$'\n\n'
for cell in easy_easy easy_easy_medium medium_easy medium_easy_medium; do
  if [ -f "$WORK_DIR/logs/phase0b_gcl_${cell}.DONE" ]; then
    BODY+="✅ ${cell} complete"$'\n'
  elif [ -f "$WORK_DIR/logs/phase0b_gcl_${cell}.FAILED" ]; then
    BODY+="❌ ${cell} FAILED"$'\n'
  fi
done
BODY+=$'\n''- [ ] Review TensorBoard logs'$'\n'
BODY+='- [ ] Compare GCL results to AIRL'$'\n'
BODY+='- [ ] Update README with Phase 0b results'

gh issue create \
  --title "Phase 0b GCL training finished ($SHA)" \
  --body "$BODY"

send_email "🏁 Swim-IRL Phase 0b GCL — all cells done" \
  "All requested GCL cells completed.\n\n$BODY\n\nRun ID: $RUN_ID\nBranch: $BRANCH"
log "All done."