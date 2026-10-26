@echo off
rem M2 go/no-go runner (Windows cmd). Run from C:\Research\dccnet with the dccnet conda env active:
rem     scripts\run_m2.bat
rem For each config (circlenet density dccell) and seed (0 1 2): train, predict val, sweep threshold on val,
rem predict test, evaluate test at the swept best_thr (no mAP CI: --map-ci-n 0, computed offline later). Resumable: re-run to continue. Stops at the first failure.
rem Optional knobs (environment variables, default off): M2_SEEDS, M2_CONFIGS, M2_RUNS_DIR, M2_DATA_ROOT,
rem M2_CONFIG_DIR, M2_DEVICE, M2_TRAIN_ARGS, M2_PREDICT_ARGS  (same meaning as in run_m2.sh).
setlocal EnableExtensions
cd /d "%~dp0.."

if not defined M2_PYTHON set "M2_PYTHON=python"
if not defined M2_DATA_ROOT set "M2_DATA_ROOT=data"
if not defined M2_RUNS_DIR set "M2_RUNS_DIR=runs"
if not defined M2_CONFIG_DIR set "M2_CONFIG_DIR=configs"
if not defined M2_SEEDS set "M2_SEEDS=0 1 2"
if not defined M2_CONFIGS set "M2_CONFIGS=circlenet density dccell"
set "DEVARG="
if defined M2_DEVICE set "DEVARG=--device %M2_DEVICE%"
set "SPLIT=%M2_DATA_ROOT%\splits\synthetic.json"
set "LOG=%M2_RUNS_DIR%\m2_runner.log"
if not exist "%M2_RUNS_DIR%" mkdir "%M2_RUNS_DIR%"

rem Synthetic data: (re)generate when the split or data\synthetic\VERSION is missing, or VERSION is not 2.
set "SYNVER="
if exist "%M2_DATA_ROOT%\synthetic\VERSION" set /p SYNVER=<"%M2_DATA_ROOT%\synthetic\VERSION"
set "REGEN="
if not exist "%SPLIT%" set "REGEN=1"
if not "%SYNVER%"=="2" set "REGEN=1"
if defined REGEN (
  call :log generate synthetic data - version 2, found "%SYNVER%"
  "%M2_PYTHON%" scripts\make_synthetic.py --out "%M2_DATA_ROOT%\synthetic" --data-root "%M2_DATA_ROOT%" --seed 0 || goto :fail
)

for %%C in (%M2_CONFIGS%) do (
  for %%S in (%M2_SEEDS%) do (
    call :one %%C %%S || goto :fail
  )
)
call :log ALL DONE
exit /b 0

:fail
call :log FAILED - see the messages above. Fix the cause and re-run scripts\run_m2.bat to resume.
exit /b 1

:log
echo %date% %time% %*
>>"%LOG%" echo %date% %time% %*
exit /b 0

:one
set "TAG=exp_m2_%1_s%2"
set "RUN=%M2_RUNS_DIR%\%TAG%"
if exist "%RUN%\summary_test.json" (
  call :log %TAG%: done already, skipping
  exit /b 0
)
set "ABORT=%TAG%_aborted_%RANDOM%"
if exist "%RUN%\best.pt" if not exist "%RUN%\summary.json" (
  call :log %TAG%: aborted earlier run found, renaming folder to %ABORT% and training fresh
  ren "%RUN%" "%ABORT%" || exit /b 1
)
if exist "%RUN%\summary.json" (
  call :log %TAG%: summary.json exists, skipping train
) else (
  call :log %TAG%: train
  "%M2_PYTHON%" scripts\train.py --config "%M2_CONFIG_DIR%\exp_m2_%1.yaml" --seed %2 --data-root "%M2_DATA_ROOT%" --runs-dir "%M2_RUNS_DIR%" %DEVARG% %M2_TRAIN_ARGS% || exit /b 1
)
call :log %TAG%: predict val
"%M2_PYTHON%" scripts\predict.py --run "%RUN%" --split "%SPLIT%" --subset val --data-root "%M2_DATA_ROOT%" --out "%RUN%\preds_val.json" %DEVARG% %M2_PREDICT_ARGS% || exit /b 1
call :log %TAG%: sweep threshold on val
"%M2_PYTHON%" scripts\evaluate.py --split "%SPLIT%" --subset val --preds "%RUN%\preds_val.json" --data-root "%M2_DATA_ROOT%" --out "%RUN%\sweep_val.json" --sweep-thr || exit /b 1
set "THR="
for /f "usebackq delims=" %%T in (`call "%M2_PYTHON%" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_thr'])" "%RUN%\sweep_val.json"`) do set "THR=%%T"
if not defined THR (
  call :log %TAG%: could not read best_thr from sweep_val.json
  exit /b 1
)
call :log %TAG%: predict test, best_thr=%THR%
"%M2_PYTHON%" scripts\predict.py --run "%RUN%" --split "%SPLIT%" --subset test --data-root "%M2_DATA_ROOT%" --out "%RUN%\preds_test.json" %DEVARG% %M2_PREDICT_ARGS% || exit /b 1
call :log %TAG%: evaluate test at thr %THR%
"%M2_PYTHON%" scripts\evaluate.py --split "%SPLIT%" --subset test --preds "%RUN%\preds_test.json" --data-root "%M2_DATA_ROOT%" --out "%RUN%\summary_test.json" --score-thr %THR% --map-ci-n 0 || exit /b 1
call :log %TAG%: finished
exit /b 0
