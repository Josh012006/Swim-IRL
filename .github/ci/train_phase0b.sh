#!/bin/bash
# .github/ci/train_phase0b.sh
# Phase 0b -- combined GCL + AIRL training pipeline, one script instead
# of two separate ones. Originally split into train_phase0b_gcl.sh and
# train_phase0b_airl.sh, each with its own workflow/service -- merged
# after confirming empirically that two independent workflows, each
# doing a full checkout on the SAME shared self-hosted-runner working
# directory, corrupted a training run still active from the other
# workflow (silent, untraced process death mid-training). One workflow,
# one job, one service removes the race by construction: there is no
# longer a second checkout that can happen while training is running.
#
# 2x2 grid (easy/medium agent x easy/easy_medium seed_mode) -- "hard"
# was dropped from both axes, see README Limitations.
#
# Structure mirrors NanoGoal-RL's own single .github/ci/train.sh:
# - Unique RUN_ID per invocation
# - Optional SendGrid email notifications at key milestones
# - CPU watcher (alerts if training silently crashes)
# - 4h log reporter
# - Per-cell sequential training (all GCL cells, then all AIRL cells),
#   evaluation, plot generation
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
log_prefix="[phase0b:$RUN_ID]"

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

# ── Read flags ────────────────────────────────────────────────────────────────
TRAIN_GCL_EASY_EASY=false
TRAIN_GCL_EASY_EASY_MEDIUM=false
TRAIN_GCL_MEDIUM_EASY=false
TRAIN_GCL_MEDIUM_EASY_MEDIUM=false
TRAIN_AIRL_EASY_EASY=false
TRAIN_AIRL_EASY_EASY_MEDIUM=false
TRAIN_AIRL_MEDIUM_EASY=false
TRAIN_AIRL_MEDIUM_EASY_MEDIUM=false

while IFS='=' read -r key value || [ -n "$key" ]; do
  [ -z "$key" ] && continue
  case "$key" in
    train_gcl_easy_easy)           TRAIN_GCL_EASY_EASY="$value" ;;
    train_gcl_easy_easy_medium)    TRAIN_GCL_EASY_EASY_MEDIUM="$value" ;;
    train_gcl_medium_easy)         TRAIN_GCL_MEDIUM_EASY="$value" ;;
    train_gcl_medium_easy_medium)  TRAIN_GCL_MEDIUM_EASY_MEDIUM="$value" ;;
    train_airl_easy_easy)          TRAIN_AIRL_EASY_EASY="$value" ;;
    train_airl_easy_easy_medium)   TRAIN_AIRL_EASY_EASY_MEDIUM="$value" ;;
    train_airl_medium_easy)        TRAIN_AIRL_MEDIUM_EASY="$value" ;;
    train_airl_medium_easy_medium) TRAIN_AIRL_MEDIUM_EASY_MEDIUM="$value" ;;
  esac
done < "$FLAG_FILE"

TRAINING_FAILED=false

log "Phase 0b training started. Run ID: $RUN_ID"
log "GCL flags:  easy/easy=$TRAIN_GCL_EASY_EASY  easy/easy_medium=$TRAIN_GCL_EASY_EASY_MEDIUM  medium/easy=$TRAIN_GCL_MEDIUM_EASY  medium/easy_medium=$TRAIN_GCL_MEDIUM_EASY_MEDIUM"
log "AIRL flags: easy/easy=$TRAIN_AIRL_EASY_EASY  easy/easy_medium=$TRAIN_AIRL_EASY_EASY_MEDIUM  medium/easy=$TRAIN_AIRL_MEDIUM_EASY  medium/easy_medium=$TRAIN_AIRL_MEDIUM_EASY_MEDIUM"
log "n_envs=$N_ENVS"

send_email "🚀 Swim-IRL Phase 0b training started ($SHA)" \
  "Phase 0b training started.\nRun ID: $RUN_ID\nBranch: $BRANCH\nCommit: $SHA\n\nGCL cells:\n- easy/easy=$TRAIN_GCL_EASY_EASY\n- easy/easy_medium=$TRAIN_GCL_EASY_EASY_MEDIUM\n- medium/easy=$TRAIN_GCL_MEDIUM_EASY\n- medium/easy_medium=$TRAIN_GCL_MEDIUM_EASY_MEDIUM\n\nAIRL cells:\n- easy/easy=$TRAIN_AIRL_EASY_EASY\n- easy/easy_medium=$TRAIN_AIRL_EASY_EASY_MEDIUM\n- medium/easy=$TRAIN_AIRL_MEDIUM_EASY\n- medium/easy_medium=$TRAIN_AIRL_MEDIUM_EASY_MEDIUM"

# ── CPU watcher ───────────────────────────────────────────────────────────────
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
        send_email "⚠️ Swim-IRL Phase 0b — low CPU detected" \
          "CPU below ${threshold}% for 5 consecutive minutes.\nRun ID: $RUN_ID\nCheck: tail -f $WORK_DIR/logs/phase0b_session.log"
        low_count=0
      fi
    else
      low_count=0
    fi
  done
}

# ── 4h log reporter ───────────────────────────────────────────────────────────
log_reporter() {
  local interval=$((4 * 3600))
  while true; do
    sleep $interval
    local body="Phase 0b progress — $(date -u '+%Y-%m-%d %H:%M UTC')\nRun ID: $RUN_ID\n\n"
    for logfile in "$WORK_DIR"/logs/phase0b_gcl_*.log "$WORK_DIR"/logs/phase0b_airl_*.log; do
      if [ -f "$logfile" ]; then
        body+="=== $(basename "$logfile") ===\n$(tail -50 "$logfile")\n\n"
      fi
    done
    send_email "📊 Swim-IRL Phase 0b report — $(date -u '+%H:%M UTC')" "$body"
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

# ── Helper: run one cell, for either algorithm ────────────────────────────────
run_cell() {
  local algo="$1"     # "gcl" or "airl"
  local model="$2"
  local seed_mode="$3"
  local cell="${model}_${seed_mode}"
  local logfile="$WORK_DIR/logs/phase0b_${algo}_${cell}.log"
  local module="experiments.phase0b_${algo}_training"

  log "Starting $algo cell $cell..."
  send_email "🟢 Swim-IRL $algo — ${cell} started" \
    "$algo cell ${cell} started.\nRun ID: $RUN_ID\nCommit: $SHA"

  RUN_ID="$RUN_ID" "$VENV/python" -u -m "$module" \
      --seed 0 --n-envs "$N_ENVS" --cell "${model}_${seed_mode}" \
      > "$logfile" 2>&1
  local exit_code=$?

  if [ "$exit_code" -eq 0 ]; then
    touch "$WORK_DIR/logs/phase0b_${algo}_${cell}.DONE"
    send_email "✅ Swim-IRL $algo — ${cell} complete" \
      "$algo cell ${cell} finished.\nRun ID: $RUN_ID\nCommit: $SHA"
    log "$algo cell $cell complete."
  else
    echo "exit_code=$exit_code" > "$WORK_DIR/logs/phase0b_${algo}_${cell}.FAILED"
    if [ "$exit_code" -gt 128 ]; then
      local signal_num=$((exit_code - 128))
      echo "killed by signal $signal_num (128 + $signal_num = $exit_code)" >> "$WORK_DIR/logs/phase0b_${algo}_${cell}.FAILED"
      log "$algo cell $cell FAILED -- killed by signal $signal_num (exit_code=$exit_code)"
    else
      echo "process-level failure, not a signal" >> "$WORK_DIR/logs/phase0b_${algo}_${cell}.FAILED"
      log "$algo cell $cell FAILED -- exit_code=$exit_code (not a signal)"
    fi
    send_email "❌ Swim-IRL $algo — ${cell} FAILED" \
      "$algo cell ${cell} failed with exit_code=$exit_code.\nRun ID: $RUN_ID\nCommit: $SHA\n\nLast logs:\n$(tail -50 "$logfile")"
    TRAINING_FAILED=true
  fi
}

# ── Run all requested cells: GCL first, then AIRL ─────────────────────────────
mkdir -p "$WORK_DIR/logs"

[ "$TRAIN_GCL_EASY_EASY"           = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell gcl easy easy
[ "$TRAIN_GCL_EASY_EASY_MEDIUM"    = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell gcl easy easy_medium
[ "$TRAIN_GCL_MEDIUM_EASY"         = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell gcl medium easy
[ "$TRAIN_GCL_MEDIUM_EASY_MEDIUM"  = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell gcl medium easy_medium
[ "$TRAIN_AIRL_EASY_EASY"          = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell airl easy easy
[ "$TRAIN_AIRL_EASY_EASY_MEDIUM"   = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell airl easy easy_medium
[ "$TRAIN_AIRL_MEDIUM_EASY"        = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell airl medium easy
[ "$TRAIN_AIRL_MEDIUM_EASY_MEDIUM" = "true" ] && [ "$TRAINING_FAILED" = "false" ] && run_cell airl medium easy_medium

# ── Reset flag file ───────────────────────────────────────────────────────────
log "Resetting train_phase0b.flag..."
{
  echo "train=false"
  echo "train_gcl_easy_easy=false"
  echo "train_gcl_easy_easy_medium=false"
  echo "train_gcl_medium_easy=false"
  echo "train_gcl_medium_easy_medium=false"
  echo "train_airl_easy_easy=false"
  echo "train_airl_easy_easy_medium=false"
  echo "train_airl_medium_easy=false"
  echo "train_airl_medium_easy_medium=false"
} > "$FLAG_FILE"

# ── Commit & push ─────────────────────────────────────────────────────────────
log "Committing results..."
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add experiments/results/ logs/ .github/ci/train_phase0b.flag
if [ -z "$(git status --porcelain)" ]; then
  log "No changes to commit."
else
  git commit -m "feat: Phase 0b results after commit $SHA [skip ci]"
  git push origin "$BRANCH"
  log "Pushed to branch $BRANCH."
fi

# ── Create GitHub issue ───────────────────────────────────────────────────────
log "Creating GitHub issue..."
BODY="Phase 0b training from commit: $SHA"$'\n\n'
for algo in gcl airl; do
  BODY+="=== $algo ==="$'\n'
  for cell in easy_easy easy_easy_medium medium_easy medium_easy_medium; do
    if [ -f "$WORK_DIR/logs/phase0b_${algo}_${cell}.DONE" ]; then
      BODY+="✅ ${cell} complete"$'\n'
    elif [ -f "$WORK_DIR/logs/phase0b_${algo}_${cell}.FAILED" ]; then
      BODY+="❌ ${cell} FAILED"$'\n'
    fi
  done
  BODY+=$'\n'
done
BODY+='- [ ] Review TensorBoard logs'$'\n'
BODY+='- [ ] Compare GCL results to AIRL'$'\n'
BODY+='- [ ] Update README with Phase 0b results'

gh issue create \
  --title "Phase 0b training finished ($SHA)" \
  --body "$BODY"

# ── Final notification ────────────────────────────────────────────────────────
send_email "🏁 Swim-IRL Phase 0b — all cells done" \
  "All requested cells completed.\n\n$BODY\n\nRun ID: $RUN_ID\nBranch: $BRANCH"
log "All done."