"""run_exp.sh with a stub python that records argv and emulates each family's outputs (bash only)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp.sh"

STUB = f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
if len(sys.argv) > 1 and sys.argv[1] == "-c":
    code = sys.argv[2]; sys.argv = ["-c"] + sys.argv[3:]; exec(code)
    raise SystemExit
script = Path(sys.argv[1]).name
args = sys.argv[2:]
def opt(name):
    return args[args.index(name) + 1] if name in args else None
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(script + " " + " ".join(args) + "\\n")
if script == os.environ.get("STUB_FAIL_ON"):
    raise SystemExit(1)
if script in ("train.py", "frcnn_train.py", "yolo_train.py"):
    run = Path(opt("--runs-dir")) / opt("--run-name")
    wdir = run / "weights" if script == "yolo_train.py" else run
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "best.pt").write_text("w")
    (run / "summary.json").write_text("{{}}")
elif opt("--out"):
    out = Path(opt("--out"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({{"best_thr": 0.4}}) if "--sweep-thr" in args else "{{}}")
"""


def _setup(tmp_path, splits=("pooled",)):
    stub = tmp_path / "stubpy"
    stub.write_text(STUB)
    stub.chmod(0o755)
    data = tmp_path / "data"
    (data / "splits").mkdir(parents=True)
    for s in splits:
        (data / "splits" / f"{s}.json").write_text("{}")
    env = {
        "PATH": "/usr/bin:/bin",
        "EXP_PYTHON": str(stub),
        "EXP_DATA_ROOT": str(data),
        "EXP_RUNS_DIR": str(tmp_path / "runs"),
        "STUB_CALLS": str(tmp_path / "calls.txt"),
    }
    return env


def _run(env, **extra):
    return subprocess.run(["bash", str(SCRIPT)], env={**env, **extra}, capture_output=True, text=True)


def _calls(tmp_path):
    p = tmp_path / "calls.txt"
    return p.read_text().splitlines() if p.exists() else []


def test_runner_order_and_families(tmp_path):
    env = _setup(tmp_path, ("pooled", "loso_PBC"))
    r = _run(env, EXP_SPLITS="pooled loso_PBC", EXP_ARMS="d_both yolo26m frcnn", EXP_SEEDS="0",
             EXP_DEVICE="cpu", EXP_TRAIN_ARGS="--epochs-max 1")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = _calls(tmp_path)
    trains = [c for c in calls if "_train.py" in c.split()[0] or c.startswith("train.py")]
    assert [c.split("--run-name ")[1].split()[0] for c in trains] == [
        f"p3b_{s}_{a}_s0" for s in ("pooled", "loso_PBC") for a in ("d_both", "yolo26m", "frcnn")]
    d = trains[3]
    assert d.startswith("train.py") and "--config configs/p3b_d_both.yaml" in d
    assert "--split splits/loso_PBC.json" in d and "--device cpu" in d and "--epochs-max 1" in d
    y = trains[4]
    assert y.startswith("yolo_train.py") and "--model yolo26m.pt" in y and "--device" not in y
    assert "--epochs-max 1" in y
    f = trains[5]
    assert f.startswith("frcnn_train.py") and "--device cpu" in f
    runs = tmp_path / "runs"
    ypred = [c for c in calls if c.startswith("yolo_predict.py")]
    assert len(ypred) == 4 and "--device" not in ypred[0]
    assert f"--weights {runs}/p3b_pooled_yolo26m_s0/weights/best.pt" in ypred[0]
    fpred = [c for c in calls if c.startswith("frcnn_predict.py")]
    assert f"--run {runs}/p3b_pooled_frcnn_s0" in fpred[0] and "--device cpu" in fpred[0]
    ev = [c for c in calls if c.startswith("evaluate.py")]
    assert "--sweep-thr" in ev[0] and "--score-thr 0.4 --map-ci-n 0" in ev[1]
    assert (runs / "p3b_loso_PBC_frcnn_s0" / "summary_test.json").exists()
    assert (runs / "exp_runner.log").exists()
    # the dccnet predict gets the full split path
    dp = [c for c in calls if c.startswith("predict.py")][0]
    assert f"--split {tmp_path}/data/splits/pooled.json" in dp


def test_runner_skips_finished_and_renames_aborted(tmp_path):
    env = _setup(tmp_path)
    runs = tmp_path / "runs"
    done = runs / "p3b_pooled_d_both_s0"
    done.mkdir(parents=True)
    (done / "summary_test.json").write_text("{}")
    ab = runs / "p3b_pooled_yolo26m_s0"
    (ab / "weights").mkdir(parents=True)
    ab2 = runs / "p3b_pooled_circlenet_s0"
    ab2.mkdir()
    (ab2 / "best.pt").write_text("x")
    r = _run(env, EXP_ARMS="d_both yolo26m circlenet", EXP_SEEDS="0")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = _calls(tmp_path)
    assert not any("p3b_pooled_d_both_s0" in c for c in calls)
    assert len(list(runs.glob("p3b_pooled_yolo26m_s0_aborted_*"))) == 1
    assert len(list(runs.glob("p3b_pooled_circlenet_s0_aborted_*"))) == 1
    assert sum(c.startswith("yolo_train.py") for c in calls) == 1
    assert sum(c.startswith("train.py") for c in calls) == 1
    assert (ab / "summary_test.json").exists()


def test_runner_stops_on_failure(tmp_path):
    env = _setup(tmp_path)
    r = _run(env, EXP_ARMS="d_both frcnn", EXP_SEEDS="0 1", STUB_FAIL_ON="train.py")
    assert r.returncode != 0
    assert len(_calls(tmp_path)) == 1


def test_runner_fails_on_missing_split(tmp_path):
    env = _setup(tmp_path)
    r = _run(env, EXP_SPLITS="loso_BCCD", EXP_ARMS="d_both", EXP_SEEDS="0")
    assert r.returncode != 0
    assert _calls(tmp_path) == []
    assert "loso_BCCD.json" in (tmp_path / "runs" / "exp_runner.log").read_text()
