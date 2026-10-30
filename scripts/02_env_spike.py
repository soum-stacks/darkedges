#!/usr/bin/env python3
"""T0.8 environment spike: run one task's tests before and after the gold patch, untraced and traced.

Run in a Kaggle CPU notebook that has the competition attached and this repo copied to
/kaggle/working/darkedges. Example (notebook cell):

    !python /kaggle/working/darkedges/scripts/02_env_spike.py requests_6644

What it does, per phase:
  pre : fresh snapshot, apply test_patch              -> the new tests must FAIL
  post: fresh snapshot, apply gold patch + test_patch -> the same tests must PASS
For each phase it times pytest untraced and under tracelens.run, writes the trace to
/tmp/<id>.<phase>.json and appends one JSON line to the log (default measurements.jsonl).

Origin: this is the notebook cell that passed on requests_6644 on 2026-10-30 (pre failed, post
passed, overhead 1.1-1.2x), turned into a script. The script form itself was only checked with a stub
tracer, not on Kaggle. If a task fails for a missing module, try --keep-anyio (FastAPI tests use
anyio markers) and note the missing module in docs/env_recipe.md.

Traces are derived from competition data: keep them out of Git.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import warnings
from pathlib import Path

# tarfile.extractall warns about the Python 3.14 default filter; harmless here.
warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_DATA = "/kaggle/input/competitions/gemma-4-developer-agent"
DEFAULT_SRC = str(Path(__file__).resolve().parent.parent / "src")
DEFAULT_LOG = ("/kaggle/working/measurements.jsonl" if os.path.isdir("/kaggle/working")
               else "measurements.jsonl")


def log(path, **kw):
    kw["ts_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(kw)


def find_task(tasks_path, iid):
    with open(tasks_path) as f:
        for line in f:
            row = json.loads(line)
            if row["instance_id"] == iid:
                return row
    sys.exit(f"instance {iid} not found in {tasks_path}")


def test_selection(test_patch, keep_anyio, whole_files):
    """Test files touched by test_patch, narrowed with -k to added or changed test names."""
    files = sorted(set(re.findall(r"^\+\+\+ b/(\S+)", test_patch, re.M)))
    names = sorted(set(re.findall(r"^\+\s*(?:async )?def (test\w+)", test_patch, re.M))
                   | set(re.findall(r"^@@.*@@.*def (test\w+)", test_patch, re.M)))
    args = [*files, "-p", "no:cacheprovider", "-q"]
    if not keep_anyio:
        args += ["-p", "no:anyio"]
    if names and not whole_files:
        args += ["-k", " or ".join(names)]
    return files, names, args


def fresh_tree(work, iid, tag, snap, patches):
    """Extract the snapshot into a clean dir and apply patches in order. Returns the repo root."""
    d = Path(work) / f"{iid}_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    with tarfile.open(snap) as t:
        t.extractall(d)
    kids = list(d.iterdir())
    if len(kids) == 1 and kids[0].is_dir():
        d = kids[0]
    for i, patch in enumerate(patches):
        pf = Path(work) / f"{iid}_{tag}_{i}.diff"
        pf.write_text(patch if patch.endswith("\n") else patch + "\n")
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", str(pf)],
                           cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"git apply failed ({tag}, patch {i}):\n{r.stderr}")
    return str(d)


def run(cmd, cwd, src, timeout):
    env = dict(os.environ, PYTHONPATH=f"{cwd}/src:{cwd}:{src}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return r, round(time.time() - t0, 2)


def tail(r, n):
    return "\n".join((r.stdout + r.stderr).splitlines()[-n:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("instance_id", help="e.g. requests_6644")
    ap.add_argument("--data", default=DEFAULT_DATA, help="competition folder (tasks.jsonl, snapshots/)")
    ap.add_argument("--src", default=DEFAULT_SRC, help="folder containing the tracelens package")
    ap.add_argument("--work", default="/tmp/w")
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--keep-anyio", action="store_true", help="do not pass -p no:anyio")
    ap.add_argument("--whole-files", action="store_true", help="run whole test files, not just new tests")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    iid = a.instance_id
    task = find_task(Path(a.data) / "tasks.jsonl", iid)
    snap = Path(a.data) / "snapshots" / f"{iid}.tgz"
    if not snap.exists():
        sys.exit(f"snapshot not found: {snap}")
    files, names, args = test_selection(task["test_patch"], a.keep_anyio, a.whole_files)
    print("test files:", files, "| tests:", names)

    outcomes = {}
    for phase, patches in [("pre", [task["test_patch"]]), ("post", [task["patch"], task["test_patch"]])]:
        root = fresh_tree(a.work, iid, phase, str(snap), patches)
        r0, t_plain = run([sys.executable, "-m", "pytest", *args], root, a.src, a.timeout)
        print(f"--- {phase} untraced {t_plain}s rc={r0.returncode}\n{tail(r0, 12)}")
        out = f"/tmp/{iid}.{phase}.json"
        if os.path.exists(out):
            os.remove(out)
        r1, t_traced = run([sys.executable, "-m", "tracelens.run", "--root", root, "--out", out,
                            "--", "pytest", *args], root, a.src, a.timeout)
        if not os.path.exists(out):
            print("TRACE FAILED:\n" + tail(r1, 20))
            log(a.log, step="T0.8", instance=iid, phase=phase, ok=False, untraced_s=t_plain)
            continue
        dj = json.load(open(out))
        outcomes[phase] = {k: v["outcome"] for k, v in dj["tests"].items()}
        log(a.log, step="T0.8", instance=iid, phase=phase, untraced_s=t_plain, traced_s=t_traced,
            overhead=round(t_traced / max(t_plain, 0.01), 1), exit_code=dj["exit_code"],
            backend=dj["backend"], outcomes=outcomes[phase],
            n_nodes=len(dj["nodes"]), n_edges=len(dj["edges"]))

    pre_failed = {k for k, v in outcomes.get("pre", {}).items() if v == "failed"}
    post_ok = bool(pre_failed) and all(outcomes.get("post", {}).get(k) == "passed" for k in pre_failed)
    verdict = "SPIKE OK" if post_ok else "CHECK: need >=1 test failing pre and passing post"
    print(f"\n{verdict} ({iid}; failed pre: {sorted(pre_failed)})")
    log(a.log, step="T0.8_verdict", instance=iid, ok=post_ok)
    return 0 if post_ok else 1


if __name__ == "__main__":
    sys.exit(main())