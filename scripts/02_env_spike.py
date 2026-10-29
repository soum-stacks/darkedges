"""UNTESTED TEMPLATE (needs the competition data; adapt after reading HARNESS_README and sandbox/setup.py).
Goal for Day 0: ONE task end to end: extract snapshot -> install offline -> apply test_patch -> run fail-to-pass tests under tracelens.
Record whatever works in docs/env_recipe.md. Prefer the harness helpers (swegemma sandbox/verification) if they are easier."""
import json, os, re, subprocess, sys, tarfile, tempfile

DATA = os.environ.get("DATA", "/kaggle/input/competitions/gemma-4-developer-agent")  # adjust
TASK_ID = os.environ.get("TASK_ID", "requests_7205")                                  # pick a small repo first
TRACELENS_SRC = os.environ.get("TRACELENS_SRC", "/kaggle/working/darkedges/src")

task = next(json.loads(l) for l in open(f"{DATA}/tasks.jsonl") if json.loads(l)["instance_id"] == TASK_ID)
work = tempfile.mkdtemp(prefix=TASK_ID + "_")
with tarfile.open(f"{DATA}/snapshots/{TASK_ID}.tgz") as tf:
    tf.extractall(work)
repo = work if os.path.isdir(f"{work}/.git") else next(os.path.join(work, d) for d in os.listdir(work) if os.path.isdir(f"{work}/{d}/.git"))
print("repo:", repo)

def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    print("$", cmd, "->", r.returncode); print((r.stdout + r.stderr)[-1500:]); return r

open(f"{work}/test.patch", "w").write(task["test_patch"])
sh(f"cd {repo} && git apply {work}/test.patch")
# TODO 1: offline install. Look at {DATA}/sandbox/setup.py; it resolves wheels from /wheels and does an editable install.
#         Try: ln -s {DATA}/wheels /wheels && python {DATA}/sandbox/setup.py   (run with cwd=repo), or pip --no-index --find-links.
# TODO 2: derive the fail-to-pass test files from test_patch:
files = sorted(set(re.findall(r"^\+\+\+ b/(\S+)", task["test_patch"], flags=re.M)))
tests = [f for f in files if re.search(r"(^|/)test_.*\.py$|_test\.py$", f)]
print("test files:", tests)
sh(f"cd {repo} && PYTHONPATH={TRACELENS_SRC} python -m tracelens.run --root {repo} --out {work}/trace.json "
   f"-- pytest {' '.join(tests)} -p no:anyio -q -x")
sh(f"python -m tracelens.summarize {work}/trace.json 2000", env=dict(os.environ, PYTHONPATH=TRACELENS_SRC))
