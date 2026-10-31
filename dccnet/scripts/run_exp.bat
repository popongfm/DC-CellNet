@echo off
rem P3b generic experiment runner (Windows cmd; twin of run_exp.sh). Run from C:\Research\dccnet with the dccnet env active:
rem     scripts\run_exp.bat
rem For each split, arm and seed: train, predict val, sweep threshold on val, predict test, evaluate test at the swept
rem best_thr (no mAP CI: --map-ci-n 0, computed offline in summarize_loso.py). Run dir: RUNS\PREFIX_split_arm_sSEED.
rem Family by arm name: yolo* = Ultralytics, frcnn = Faster R-CNN, else DC-CellNet with CONFIG_DIR\PREFIX_arm.yaml.
rem Resumable: re-run to continue. A killed run - best.pt or weights\ without summary.json - is renamed aside and retrained.
rem Extra eval splits (EXP_EXTRA_SPLITS): after each run's normal stage, also for finished runs (predict only, no retraining),
rem predict DATA\splits\x.json (subset test) to RUN\preds_x.json and evaluate at the run's best_thr to RUN\summary_x.json (skipped if present).
rem Splits are never regenerated: a missing DATA\splits\split.json is a failure. Stops at the first failure.
rem Environment variables: EXP_PYTHON EXP_DATA_ROOT EXP_RUNS_DIR EXP_CONFIG_DIR EXP_PREFIX EXP_SPLITS EXP_ARMS EXP_SEEDS
rem EXP_DEVICE EXP_TRAIN_ARGS EXP_PREDICT_ARGS EXP_EXTRA_SPLITS  (same meaning and defaults as in run_exp.sh).
setlocal EnableExtensions
cd /d "%~dp0.."

if not defined EXP_PYTHON set "EXP_PYTHON=python"
if not defined EXP_DATA_ROOT set "EXP_DATA_ROOT=data"
if not defined EXP_RUNS_DIR set "EXP_RUNS_DIR=runs"
if not defined EXP_CONFIG_DIR set "EXP_CONFIG_DIR=configs"
if not defined EXP_PREFIX set "EXP_PREFIX=p3b"
if not defined EXP_SPLITS set "EXP_SPLITS=pooled"
if not defined EXP_ARMS set "EXP_ARMS=d_both d_sigma circlenet yolo26m frcnn"
if not defined EXP_SEEDS set "EXP_SEEDS=0 1 2"
set "DEVARG="
if defined EXP_DEVICE set "DEVARG=--device %EXP_DEVICE%"
set "LOG=%EXP_RUNS_DIR%\exp_runner.log"
if not exist "%EXP_RUNS_DIR%" mkdir "%EXP_RUNS_DIR%"

for %%P in (%EXP_SPLITS% %EXP_EXTRA_SPLITS%) do (
  if not exist "%EXP_DATA_ROOT%\splits\%%P.json" (
    call :log missing split file %EXP_DATA_ROOT%\splits\%%P.json - run prepare_data.py first, not regenerating
    goto :fail
  )
)

for %%P in (%EXP_SPLITS%) do (
  for %%A in (%EXP_ARMS%) do (
    for %%S in (%EXP_SEEDS%) do (
      call :one %%P %%A %%S || goto :fail
    )
  )
)
call :log ALL DONE
exit /b 0

:fail
call :log FAILED - see the messages above. Fix the cause and re-run scripts\run_exp.bat to resume.
exit /b 1

:log
echo %date% %time% %*
>>"%LOG%" echo %date% %time% %*
exit /b 0

:one
set "TAG=%EXP_PREFIX%_%1_%2_s%3"
set "RUN=%EXP_RUNS_DIR%\%TAG%"
set "SPLIT=%EXP_DATA_ROOT%\splits\%1.json"
set "ARM=%2"
set "FAM=dccnet"
if /i "%ARM:~0,4%"=="yolo" set "FAM=yolo"
if /i "%ARM%"=="frcnn" set "FAM=frcnn"
if exist "%RUN%\summary_test.json" (
  call :log %TAG%: done already, skipping
  goto :extra
)
set "ABORT=%TAG%_aborted_%RANDOM%%RANDOM%"
set "HAVEW="
if exist "%RUN%\best.pt" set "HAVEW=1"
if exist "%RUN%\weights\" set "HAVEW=1"
if defined HAVEW if not exist "%RUN%\summary.json" (
  call :log %TAG%: aborted earlier run found, renaming folder to %ABORT% and training fresh
  ren "%RUN%" "%ABORT%" || exit /b 1
)
if exist "%RUN%\summary.json" (
  call :log %TAG%: summary.json exists, skipping train
  goto :predict
)
call :log %TAG%: train, %FAM%
if "%FAM%"=="dccnet" (
  "%EXP_PYTHON%" scripts\train.py --config "%EXP_CONFIG_DIR%\%EXP_PREFIX%_%ARM%.yaml" --split "splits/%1.json" --seed %3 --data-root "%EXP_DATA_ROOT%" --runs-dir "%EXP_RUNS_DIR%" --run-name "%TAG%" %DEVARG% %EXP_TRAIN_ARGS% || exit /b 1
)
if "%FAM%"=="yolo" (
  "%EXP_PYTHON%" scripts\baselines\yolo_train.py --split "splits/%1.json" --data-root "%EXP_DATA_ROOT%" --runs-dir "%EXP_RUNS_DIR%" --run-name "%TAG%" --seed %3 --model "%ARM%.pt" %EXP_TRAIN_ARGS% || exit /b 1
)
if "%FAM%"=="frcnn" (
  "%EXP_PYTHON%" scripts\baselines\frcnn_train.py --split "splits/%1.json" --data-root "%EXP_DATA_ROOT%" --runs-dir "%EXP_RUNS_DIR%" --run-name "%TAG%" --seed %3 %DEVARG% %EXP_TRAIN_ARGS% || exit /b 1
)

:predict
call :log %TAG%: predict val
call :pred val "%SPLIT%" "%RUN%\preds_val.json" || exit /b 1
call :log %TAG%: sweep threshold on val
"%EXP_PYTHON%" scripts\evaluate.py --split "%SPLIT%" --subset val --preds "%RUN%\preds_val.json" --data-root "%EXP_DATA_ROOT%" --out "%RUN%\sweep_val.json" --sweep-thr || exit /b 1
set "THR="
for /f "usebackq delims=" %%T in (`call "%EXP_PYTHON%" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_thr'])" "%RUN%\sweep_val.json"`) do set "THR=%%T"
if not defined THR (
  call :log %TAG%: could not read best_thr from sweep_val.json
  exit /b 1
)
call :log %TAG%: predict test, best_thr=%THR%
call :pred test "%SPLIT%" "%RUN%\preds_test.json" || exit /b 1
call :log %TAG%: evaluate test at thr %THR%
"%EXP_PYTHON%" scripts\evaluate.py --split "%SPLIT%" --subset test --preds "%RUN%\preds_test.json" --data-root "%EXP_DATA_ROOT%" --out "%RUN%\summary_test.json" --score-thr %THR% --map-ci-n 0 || exit /b 1
call :log %TAG%: finished

:extra
for %%X in (%EXP_EXTRA_SPLITS%) do (
  call :extra1 %%X || exit /b 1
)
exit /b 0

:extra1
rem %1 = extra split name; uses TAG RUN FAM from :one. Predict only (no retraining), scored at the run's own best_thr
if exist "%RUN%\summary_%1.json" (
  call :log %TAG%: %1 done already, skipping
  exit /b 0
)
set "XSPLIT=%EXP_DATA_ROOT%\splits\%1.json"
set "THR="
for /f "usebackq delims=" %%T in (`call "%EXP_PYTHON%" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_thr'])" "%RUN%\sweep_val.json"`) do set "THR=%%T"
if not defined THR (
  call :log %TAG%: could not read best_thr from sweep_val.json
  exit /b 1
)
call :log %TAG%: predict %1, best_thr=%THR%
call :pred test "%XSPLIT%" "%RUN%\preds_%1.json" || exit /b 1
call :log %TAG%: evaluate %1 at thr %THR%
"%EXP_PYTHON%" scripts\evaluate.py --split "%XSPLIT%" --subset test --preds "%RUN%\preds_%1.json" --data-root "%EXP_DATA_ROOT%" --out "%RUN%\summary_%1.json" --score-thr %THR% --map-ci-n 0 || exit /b 1
exit /b 0

:pred
rem %1 = val or test; %2 = split json; %3 = output json; uses RUN FAM from :one (Ultralytics predict gets no --device: default 0)
if "%FAM%"=="dccnet" (
  "%EXP_PYTHON%" scripts\predict.py --run "%RUN%" --split %2 --subset %1 --data-root "%EXP_DATA_ROOT%" --out %3 %DEVARG% %EXP_PREDICT_ARGS% || exit /b 1
)
if "%FAM%"=="yolo" (
  "%EXP_PYTHON%" scripts\baselines\yolo_predict.py --weights "%RUN%\weights\best.pt" --split %2 --subset %1 --data-root "%EXP_DATA_ROOT%" --out %3 %EXP_PREDICT_ARGS% || exit /b 1
)
if "%FAM%"=="frcnn" (
  "%EXP_PYTHON%" scripts\baselines\frcnn_predict.py --run "%RUN%" --split %2 --subset %1 --data-root "%EXP_DATA_ROOT%" --out %3 %DEVARG% %EXP_PREDICT_ARGS% || exit /b 1
)
exit /b 0
