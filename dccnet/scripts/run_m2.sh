#!/usr/bin/env bash
# M2 go/no-go runner (bash twin of run_m2.bat). Run from the dccnet/ directory.
#
# For each config in (circlenet density dccell) and seed in (0 1 2): train -> predict val -> sweep threshold on val ->
# predict test -> evaluate test at the swept best_thr. Resumable: a run with runs/exp_m2_<cfg>_s<seed>/summary_test.json
# is skipped, a run with summary.json skips training. Stops at the first failing command.
#
# Optional knobs (environment variables, all default-off / default to the real experiment):
#   M2_PYTHON (python)  M2_DATA_ROOT (data)  M2_RUNS_DIR (runs)  M2_CONFIG_DIR (configs)
#   M2_SEEDS ("0 1 2")  M2_DEVICE (train/predict default)  M2_TRAIN_ARGS (extra args for train.py, e.g. "--epochs-max 2")
#   M2_PREDICT_ARGS (extra args for predict.py, e.g. "--batch 2")  M2_CONFIGS ("circlenet density dccell")
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${M2_PYTHON:-python}"
DATA="${M2_DATA_ROOT:-data}"
RUNS="${M2_RUNS_DIR:-runs}"
CFGDIR="${M2_CONFIG_DIR:-configs}"
SEEDS="${M2_SEEDS:-0 1 2}"
CONFIGS="${M2_CONFIGS:-circlenet density dccell}"
DEV=()
[ -n "${M2_DEVICE:-}" ] && DEV=(--device "$M2_DEVICE")
# shellcheck disable=SC2206
TRAIN_ARGS=(${M2_TRAIN_ARGS:-})
# shellcheck disable=SC2206
PRED_ARGS=(${M2_PREDICT_ARGS:-})
SPLIT_REL="splits/synthetic.json"
SPLIT="$DATA/$SPLIT_REL"

mkdir -p "$RUNS"
LOG="$RUNS/m2_runner.log"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
trap 'log "FAILED (exit $?) at: $BASH_COMMAND"' ERR

if [ ! -f "$SPLIT" ]; then
  log "generate synthetic data -> $SPLIT"
  "$PY" scripts/make_synthetic.py --out "$DATA/synthetic" --data-root "$DATA" --seed 0
fi

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    run="$RUNS/exp_m2_${cfg}_s${seed}"
    tag="exp_m2_${cfg}_s${seed}"
    if [ -f "$run/summary_test.json" ]; then
      log "$tag: done already, skipping"
      continue
    fi
    if [ -f "$run/best.pt" ] && [ ! -f "$run/summary.json" ]; then
      aborted="${run}_aborted_$(date +%s)"
      log "$tag: aborted earlier run found, moving it to $aborted and training fresh"
      mv "$run" "$aborted"
    fi
    if [ ! -f "$run/summary.json" ]; then
      log "$tag: train"
      "$PY" scripts/train.py --config "$CFGDIR/exp_m2_${cfg}.yaml" --seed "$seed" --data-root "$DATA" \
        --runs-dir "$RUNS" ${DEV[@]+"${DEV[@]}"} ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
    else
      log "$tag: summary.json exists, skipping train"
    fi
    log "$tag: predict val"
    "$PY" scripts/predict.py --run "$run" --split "$SPLIT" --subset val --data-root "$DATA" \
      --out "$run/preds_val.json" ${DEV[@]+"${DEV[@]}"} ${PRED_ARGS[@]+"${PRED_ARGS[@]}"}
    log "$tag: sweep threshold on val"
    "$PY" scripts/evaluate.py --split "$SPLIT" --subset val --preds "$run/preds_val.json" --data-root "$DATA" \
      --out "$run/sweep_val.json" --sweep-thr
    thr="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_thr'])" "$run/sweep_val.json")"
    log "$tag: predict test (best_thr=$thr)"
    "$PY" scripts/predict.py --run "$run" --split "$SPLIT" --subset test --data-root "$DATA" \
      --out "$run/preds_test.json" ${DEV[@]+"${DEV[@]}"} ${PRED_ARGS[@]+"${PRED_ARGS[@]}"}
    log "$tag: evaluate test at thr $thr"
    "$PY" scripts/evaluate.py --split "$SPLIT" --subset test --preds "$run/preds_test.json" --data-root "$DATA" \
      --out "$run/summary_test.json" --score-thr "$thr" --map-ci-n 200
    log "$tag: finished"
  done
done
log "ALL DONE"
