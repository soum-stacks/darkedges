"""Phase 1 tracing pipeline (runs on Kaggle CPU; also usable anywhere the competition data exists).

For every task: extract snapshot -> apply patches -> build offline env (envbuild) -> run the test
files touched by test_patch under `python -m tracelens.run` in two phases:
    pre  = base + test_patch
    post = base + gold patch + test_patch
Outputs (formats = Handoff Pack Section 7):
    <out>/traces/<id>.<pre|post>.json     tracelens trace (Section 7.2)
    <out>/traces/_status.jsonl            {"instance_id","phase","ok","tests_failed","tests_passed","seconds","error"}
    <out>/traces/_env.jsonl               {"instance_id","n_wheels","env_seconds","notes"}      (private log)
    <out>/results/gold_nodes.jsonl        {"instance_id","gold_node_ids","n_hunks","unmapped_hunks","unmatched"}
    <out>/traces/_valid.jsonl             written by validity_report()
Resumable: a (task, phase) with a _status row is skipped unless retry_failed and ok == false.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

from . import envbuild, hunks

PYTEST_ARGS = ["-q", "-p", "no:cacheprovider", "-p", "darkedges.pytest_shims", "--continue-on-collection-errors"]
SHIMS_DIR = Path(__file__).resolve().parent / "shims"   # stand-ins for packages absent from the wheelhouse (last on PYTHONPATH)
TEST_TIMEOUT = 900                                 # seconds per (task, phase) run
SRC_DIR = Path(__file__).resolve().parents[1]      # directory that holds tracelens/ and darkedges/


# ------------------------------------------------------------------ locating competition data
def locate(comp_root=None) -> dict:
    """Find tasks.jsonl, snapshots/, graphs/, wheels/, docker/ under the competition input dir."""
    roots = [Path(comp_root)] if comp_root else sorted(Path("/kaggle/input").glob("*")) + \
        sorted(Path("/kaggle/input/competitions").glob("*"))
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if len(p.relative_to(root).parts) > 3:
                continue
            n = p.name
            if n == "tasks.jsonl":
                found.setdefault("tasks", p)
            elif p.is_dir() and n in ("snapshots", "graphs", "wheels", "docker", "embeddings"):
                found.setdefault(n, p)
        if {"tasks", "snapshots", "graphs", "wheels", "docker"} <= set(found):
            break
    missing = {"tasks", "snapshots", "graphs", "wheels", "docker"} - set(found)
    if missing:
        raise FileNotFoundError(f"not found under /kaggle/input: {sorted(missing)}; found={found}")
    return found


def load_tasks(path) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def repo_short(task: dict) -> str:
    return task["repo"].split("/")[-1].lower()


def graph_path(graphs_dir, task: dict):
    g = Path(graphs_dir)
    for c in (g / f"{task['instance_id']}.json", g / f"{repo_short(task)}_{task['base_commit']}.json"):
        if c.is_file() and c.stat().st_size > 100:
            return c
    for c in g.glob(f"*{task['base_commit']}*.json"):
        if c.stat().st_size > 100:
            return c
    return None


def graph_node_ids(path) -> set:
    return {n["id"] for n in json.loads(Path(path).read_text()).get("nodes", [])}


# ------------------------------------------------------------------ workdir + patches
def extract_snapshot(tgz, dest) -> Path:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(tgz) as t:
        t.extractall(dest, filter="data") if sys.version_info >= (3, 12) else t.extractall(dest)
    kids = [p for p in dest.iterdir() if p.name != ".git"]
    if len(kids) == 1 and kids[0].is_dir() and not any((dest / f).exists() for f in ("setup.py", "pyproject.toml", "setup.cfg")):
        return kids[0]                               # tarball with a single top-level directory
    return dest


def apply_patch(repo: Path, patch_text: str, name: str):
    """-> (ok, message). git apply first, GNU patch as fallback."""
    if not patch_text.strip():
        return True, "empty"
    pf = repo.parent / f"_{name}.diff"
    pf.write_text(patch_text if patch_text.endswith("\n") else patch_text + "\n")
    r = subprocess.run(["git", "apply", "--whitespace=nowarn", str(pf)], cwd=repo, capture_output=True, text=True)
    if r.returncode == 0:
        return True, "git apply"
    r2 = subprocess.run(["patch", "-p1", "-f", "-i", str(pf)], cwd=repo, capture_output=True, text=True)
    if r2.returncode == 0:
        return True, "patch -p1"
    return False, (r.stderr or r.stdout)[-200:] + " | " + (r2.stdout or r2.stderr)[-200:]


def test_files(test_patch: str, repo: Path) -> list:
    out = []
    for m in re.finditer(r"^\+\+\+ b/(\S+)", test_patch, flags=re.M):
        rel = m.group(1)
        base = rel.rsplit("/", 1)[-1]
        if rel.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py")) and (repo / rel).is_file():
            out.append(rel)
    return sorted(set(out))


# ------------------------------------------------------------------ one run
def _counts(trace_path: Path):
    try:
        d = json.loads(trace_path.read_text())
    except Exception as e:
        return None, 0, 0, f"unreadable trace: {e}"
    tests = d.get("tests") or {}
    failed = sum(1 for t in tests.values() if t.get("outcome") == "failed")
    passed = sum(1 for t in tests.values() if t.get("outcome") == "passed")
    return d, failed, passed, None if tests else "no tests recorded (collection error or plugin inactive)"


def run_tests(repo: Path, deps: Path, files: list, trace_out: Path, timeout=TEST_TIMEOUT, pytest_args=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), str(deps), str(repo), str(repo / "src"), str(SHIMS_DIR)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    cmd = [sys.executable, "-m", "tracelens.run", "--root", str(repo), "--out", str(trace_out), "--",
           "pytest", *files, *(pytest_args if pytest_args is not None else PYTEST_ARGS)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + "\n" + p.stderr)[-600:], round(time.time() - t0, 2)
    except subprocess.TimeoutExpired:
        return None, "timeout", round(time.time() - t0, 2)


def _append(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def done_phases(status_path: Path, retry_failed=False) -> set:
    done = set()
    if status_path.is_file():
        for l in status_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r["ok"] or not retry_failed:
                    done.add((r["instance_id"], r["phase"]))
    return done


def run_task(task: dict, loc: dict, out: Path, work: Path, wheelhouse, done=frozenset(), pytest_args=None,
             keep_work=False) -> list:
    """Trace one task (both phases) and compute gold nodes. Returns the status rows written."""
    iid = task["instance_id"]
    tdir, rdir = Path(out) / "traces", Path(out) / "results"
    tdir.mkdir(parents=True, exist_ok=True); rdir.mkdir(parents=True, exist_ok=True)
    phases = [p for p in ("pre", "post") if (iid, p) not in done]
    gold_done = (rdir / "gold_nodes.jsonl").is_file() and f'"{iid}"' in (rdir / "gold_nodes.jsonl").read_text()
    if not phases and gold_done:
        return []
    tw = Path(work) / iid
    rows = []

    def fail(phase, msg, secs=0.0):
        row = {"instance_id": iid, "phase": phase, "ok": False, "tests_failed": 0, "tests_passed": 0,
               "seconds": secs, "error": msg}
        _append(tdir / "_status.jsonl", row); rows.append(row)

    try:
        snap = Path(loc["snapshots"]) / f"{iid}.tgz"
        # ---- pre checkout (also used for env build and gold nodes)
        pre = extract_snapshot(snap, tw / "pre")
        ok, msg = apply_patch(pre, task["test_patch"], "test_patch")
        if not ok:
            for p in phases:
                fail(p, "test_patch failed: " + msg)
            return rows
        # ---- gold nodes (base files; test_patch never touches them)
        if not gold_done:
            gp = graph_path(loc["graphs"], task)
            if gp is None:
                _append(rdir / "gold_nodes.jsonl", {"instance_id": iid, "gold_node_ids": [], "n_hunks": 0,
                                                    "unmapped_hunks": 0, "unmatched": [], "error": "no graph"})
            else:
                res = hunks.gold_nodes_detailed(task["patch"], pre, graph_node_ids(gp))
                _append(rdir / "gold_nodes.jsonl", {"instance_id": iid, **res})
        if not phases:
            return rows
        # ---- env
        info = envbuild.build_env(pre, tw / "deps", wheelhouse, docker_dir=loc["docker"], repo_short=repo_short(task))
        shims = [q.name for q in SHIMS_DIR.iterdir() if q.is_dir() and not (info.deps_dir / q.name).exists()] \
            if SHIMS_DIR.is_dir() else []
        _append(tdir / "_env.jsonl", {"instance_id": iid, "n_wheels": info.n_wheels, "shims_on_path": shims,
                                      "env_seconds": info.seconds, "notes": info.notes})
        for phase in phases:
            if phase == "pre":
                repo = pre
            else:
                repo = extract_snapshot(snap, tw / "post")
                ok1, m1 = apply_patch(repo, task["patch"], "gold")
                ok2, m2 = apply_patch(repo, task["test_patch"], "test_patch")
                if not (ok1 and ok2):
                    fail(phase, "patch failed: " + (m1 if not ok1 else m2)); continue
            files = test_files(task["test_patch"], repo)
            if not files:
                fail(phase, "no test files in test_patch"); continue
            tp = tdir / f"{iid}.{phase}.json"
            if tp.exists():
                tp.unlink()
            code, tail, secs = run_tests(repo, info.deps_dir, files, tp, pytest_args=pytest_args)
            if not tp.is_file():
                fail(phase, f"no trace written (exit {code}): {tail[-250:]}", secs); continue
            _d, nf, npass, err = _counts(tp)
            row = {"instance_id": iid, "phase": phase, "ok": err is None, "tests_failed": nf,
                   "tests_passed": npass, "seconds": secs, "error": err and f"{err}; exit {code}: {tail[-200:]}"}
            _append(tdir / "_status.jsonl", row); rows.append(row)
    except Exception as e:                                    # keep the batch alive
        for p in phases:
            if not any(r["phase"] == p for r in rows):
                fail(p, f"exception: {type(e).__name__}: {e}"[:300])
    finally:
        if not keep_work:
            shutil.rmtree(tw, ignore_errors=True)
    return rows


def run_all(loc, out, work, ids=None, shard=(0, 1), time_budget_h=11.0, retry_failed=False, prior_dir=None,
            pytest_args=None, log=print):
    out, work = Path(out), Path(work)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    if prior_dir and Path(prior_dir).is_dir():                 # resume from a previous notebook version's output
        for sub in ("traces", "results"):
            sp = Path(prior_dir) / sub
            if sp.is_dir():
                for f in sp.iterdir():
                    if not (out / sub / f.name).exists():
                        (out / sub).mkdir(parents=True, exist_ok=True); shutil.copy2(f, out / sub / f.name)
    tasks = sorted(load_tasks(loc["tasks"]), key=lambda t: t["instance_id"])
    if ids:
        tasks = [t for t in tasks if t["instance_id"] in set(ids)]
    tasks = [t for i, t in enumerate(tasks) if i % shard[1] == shard[0]]
    wh = envbuild.Wheelhouse.scan(loc["wheels"])
    done = done_phases(out / "traces" / "_status.jsonl", retry_failed)
    t0 = time.time()
    for k, t in enumerate(tasks, 1):
        if time.time() - t0 > time_budget_h * 3600:
            log(f"time budget reached after {k-1} tasks; re-run to resume"); break
        rows = run_task(t, loc, out, work, wh, done=done, pytest_args=pytest_args)
        log(f"[{k}/{len(tasks)}] {t['instance_id']}: " +
            (", ".join(f"{r['phase']} ok={r['ok']} F{r['tests_failed']}/P{r['tests_passed']} {r['seconds']}s" for r in rows) or "skipped"))
    return out


# ------------------------------------------------------------------ validity + gate
def validity_report(out, tasks_path, gate=100, log=print) -> dict:
    """Two definitions, both written to _valid.jsonl:
    strict : pre has >=1 failed test AND post has 0 failed and >=1 passed        (Handoff Addendum 1)
    f2p    : >=1 test failed in pre and passed in post (SWE-bench style; tests failing in BOTH are
             'env-limited' and listed in both_failing). POST-HOC amendment: log it in docs/amendments.md."""
    out = Path(out); tdir = out / "traces"
    st = {}
    if (tdir / "_status.jsonl").is_file():
        for l in (tdir / "_status.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l); st[(r["instance_id"], r["phase"])] = r      # last row wins
    rows = []
    for t in load_tasks(tasks_path):
        iid = t["instance_id"]; pre, post = st.get((iid, "pre")), st.get((iid, "post"))
        reason = None
        if not pre or not pre["ok"]:
            reason = "pre: " + (pre["error"] if pre else "not run")
        elif pre["tests_failed"] < 1:
            reason = "pre: no failing test"
        elif not post or not post["ok"]:
            reason = "post: " + (post["error"] if post else "not run")
        elif post["tests_failed"] > 0:
            reason = "post: failures remain"
        elif post["tests_passed"] < 1:
            reason = "post: no passing test"
        f2p, both = [], 0
        if pre and pre["ok"] and post and post["ok"]:
            a = json.loads((tdir / f"{iid}.pre.json").read_text())["tests"]
            b = json.loads((tdir / f"{iid}.post.json").read_text())["tests"]
            f2p = sorted(k for k, v in a.items() if v["outcome"] == "failed" and b.get(k, {}).get("outcome") == "passed")
            both = sum(1 for k, v in a.items() if v["outcome"] == "failed" and b.get(k, {}).get("outcome") == "failed")
        if reason is None and not f2p:
            reason = "no fail-to-pass test (failures pre do not pass post)"
        rows.append({"instance_id": iid, "repo": t["repo"], "valid": reason is None, "valid_f2p": bool(f2p),
                     "reason": reason, "f2p": f2p, "both_failing": both,
                     "pre_failed": pre and pre["tests_failed"], "post_failed": post and post["tests_failed"],
                     "post_passed": post and post["tests_passed"]})
    (tdir / "_valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = {}
    for key in ("valid", "valid_f2p"):
        by, tot = Counter(), Counter()
        for r in rows:
            tot[r["repo"]] += 1; by[r["repo"]] += r[key]
        nv = sum(r[key] for r in rows)
        log(f"[{key}] " + ", ".join(f"{k} {by[k]}/{tot[k]}" for k in sorted(tot)) +
            f"  => {nv}/{len(rows)}  gate >= {gate}: {'PASS' if nv >= gate else 'FAIL'}")
        res[key] = {"valid": nv, "total": len(rows), "per_repo": dict(by)}
    for k, n in Counter((r["reason"] or "")[:70] for r in rows if not r["valid_f2p"]).most_common(8):
        log(f"   not f2p-valid: {n:3d}  {k}")
    return res
