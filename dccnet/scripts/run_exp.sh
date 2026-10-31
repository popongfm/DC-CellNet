#!/usr/bin/env bash
# P3b generic experiment runner (bash twin of run_exp.bat). Run from anywhere; it cd's to the dccnet/ directory.
#
# For each split, arm and seed: train -> predict val -> sweep threshold on val -> predict test -> evaluate test at the
# swept best_thr (--map-ci-n 0: the mAP CI is computed offline in summarize_loso.py). Run dir:
# <runs>/<prefix>_<split>_<arm>_s<seed>. Family by arm name: yolo* -> Ultralytics (--model <arm>.pt), frcnn -> Faster R-CNN,
# anything else -> DC-CellNet with <config-dir>/<prefix>_<arm>.yaml. Resumable: summary_test.json -> skip, summary.json -> skip
# training, killed run (best.pt or weights/ but no summary.json) -> renamed aside to <tag>_aborted_<random> and retrained.
# Extra eval splits (EXP_EXTRA_SPLITS): after each run's normal stage (also for finished runs: predict only, no retraining), predict
# <data-root>/splits/<x>.json (subset test) -> <run>/preds_<x>.json and evaluate at the run's best_thr -> <run>/summary_<x>.json (skipped when present).
# Splits are never regenerated: a missing <data-root>/splits/<split>.json is a failure. Stops at the first failing command.
#
# Environment variables (defaults): EXP_PYTHON (python)  EXP_DATA_ROOT (data)  EXP_RUNS_DIR (runs)  EXP_CONFIG_DIR (configs)
#   EXP_PREFIX (p3b)  EXP_SPLITS (pooled)  EXP_ARMS ("d_both d_sigma circlenet yolo26m frcnn")  EXP_SEEDS ("0 1 2")
#   EXP_EXTRA_SPLITS (unset; extra eval splits, e.g. "overlap_bccd bccd_overlap_v2")
#   EXP_DEVICE (unset; --device for dccnet and frcnn scripts only, Ultralytics keeps its default 0)
#   EXP_TRAIN_ARGS (unset; appended to every train command, e.g. "--epochs-max 1")  EXP_PREDICT_ARGS (unset)
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PY="${EXP_PYTHON:-python}"
DATA="${EXP_DATA_ROOT:-data}"
RUNS="${EXP_RUNS_DIR:-runs}"
CFGDIR="${EXP_CONFIG_DIR:-configs}"
PREFIX="${EXP_PREFIX:-p3b}"
SPLITS="${EXP_SPLITS:-pooled}"
ARMS="${EXP_ARMS:-d_both d_sigma circlenet yolo26m frcnn}"
SEEDS="${EXP_SEEDS:-0 1 2}"
EXTRA="${EXP_EXTRA_SPLITS:-}"
DEV=()
[ -n "${EXP_DEVICE:-}" ] && DEV=(--device "$EXP_DEVICE")
# shellcheck disable=SC2206
TRAIN_ARGS=(${EXP_TRAIN_ARGS:-})
# shellcheck disable=SC2206
PRED_ARGS=(${EXP_PREDICT_ARGS:-})

mkdir -p "$RUNS"
LOG="$RUNS/exp_runner.log"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
trap 'log "FAILED (exit $?) at: $BASH_COMMAND"' ERR

# check every split file up front (never regenerated here)
for split in $SPLITS $EXTRA; do
  if [ ! -f "$DATA/splits/$split.json" ]; then
    log "missing split file $DATA/splits/$split.json (run prepare_data.py first); not regenerating"
    exit 1
  fi
done

# family predict: do_predict <subset> <split json> <out json>  (uses family, run, DEV, PRED_ARGS)
do_predict() {
  case "$family" in
    dccnet)
      "$PY" scripts/predict.py --run "$run" --split "$2" --subset "$1" --data-root "$DATA" \
        --out "$3" ${DEV[@]+"${DEV[@]}"} ${PRED_ARGS[@]+"${PRED_ARGS[@]}"} ;;
    yolo)
      "$PY" scripts/baselines/yolo_predict.py --weights "$run/weights/best.pt" --split "$2" --subset "$1" \
        --data-root "$DATA" --out "$3" ${PRED_ARGS[@]+"${PRED_ARGS[@]}"} ;;
    frcnn)
      "$PY" scripts/baselines/frcnn_predict.py --run "$run" --split "$2" --subset "$1" --data-root "$DATA" \
        --out "$3" ${DEV[@]+"${DEV[@]}"} ${PRED_ARGS[@]+"${PRED_ARGS[@]}"} ;;
  esac
}

for split in $SPLITS; do
  SPLIT="$DATA/splits/$split.json"
  for arm in $ARMS; do
    case "$arm" in
      yolo*) family=yolo ;;
      frcnn) family=frcnn ;;
      *) family=dccnet ;;
    esac
    for seed in $SEEDS; do
      tag="${PREFIX}_${split}_${arm}_s${seed}"
      run="$RUNS/$tag"
      if [ -f "$run/summary_test.json" ]; then
        log "$tag: done already, skipping"
      else
        if { [ -f "$run/best.pt" ] || [ -d "$run/weights" ]; } && [ ! -f "$run/summary.json" ]; then
          aborted="${run}_aborted_${RANDOM}${RANDOM}"
          log "$tag: aborted earlier run found, moving it to $aborted and training fresh"
          mv "$run" "$aborted"
        fi
        if [ ! -f "$run/summary.json" ]; then
          log "$tag: train ($family)"
          case "$family" in
            dccnet)
              "$PY" scripts/train.py --config "$CFGDIR/${PREFIX}_${arm}.yaml" --split "splits/$split.json" --seed "$seed" \
                --data-root "$DATA" --runs-dir "$RUNS" --run-name "$tag" ${DEV[@]+"${DEV[@]}"} ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} ;;
            yolo)
              "$PY" scripts/baselines/yolo_train.py --split "splits/$split.json" --data-root "$DATA" --runs-dir "$RUNS" \
                --run-name "$tag" --seed "$seed" --model "${arm}.pt" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} ;;
            frcnn)
              "$PY" scripts/baselines/frcnn_train.py --split "splits/$split.json" --data-root "$DATA" --runs-dir "$RUNS" \
                --run-name "$tag" --seed "$seed" ${DEV[@]+"${DEV[@]}"} ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} ;;
          esac
        else
          log "$tag: summary.json exists, skipping train"
        fi
        for subset in val test; do
          if [ "$subset" = val ]; then log "$tag: predict val"; else log "$tag: predict test (best_thr=$thr)"; fi
          do_predict "$subset" "$SPLIT" "$run/preds_${subset}.json"
          if [ "$subset" = val ]; then
            log "$tag: sweep threshold on val"
            "$PY" scripts/evaluate.py --split "$SPLIT" --subset val --preds "$run/preds_val.json" --data-root "$DATA" \
              --out "$run/sweep_val.json" --sweep-thr
            thr="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_thr'])" "$run/sweep_val.json")"
          fi
        done
        log "$tag: evaluate test at thr $thr"
        "$PY" scripts/evaluate.py --split "$SPLIT" --subset test --preds "$run/preds_test.json" --data-root "$DATA" \
          --out "$run/summary_test.json" --score-thr "$thr" --map-ci-n 0
        log "$tag: finished"
      fi
      # extra eval splits: predict only (no retraining), scored at this run's own val-swept best_thr
      for x in $EXTRA; do
        if [ -f "$run/summary_$x.json" ]; then
          log "$tag: $x done already, skipping"
          continue
        fi
        thr="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_thr'])" "$run/sweep_val.json")"
        log "$tag: predict $x (best_thr=$thr)"
        do_predict test "$DATA/splits/$x.json" "$run/preds_$x.json"
        log "$tag: evaluate $x at thr $thr"
        "$PY" scripts/evaluate.py --split "$DATA/splits/$x.json" --subset test --preds "$run/preds_$x.json" \
          --data-root "$DATA" --out "$run/summary_$x.json" --score-thr "$thr" --map-ci-n 0
      done
    done
  done
done
log "ALL DONE"
