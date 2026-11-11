#!/usr/bin/env python3
"""TraceLens agent tool: run a repro script or pytest under the runtime call tracer and print a
compact, ranked list of the repository functions that actually executed (stdlib only, offline).

Usage (from /workspace):
  python3 trace_run.py tests/_repro.py                    # trace a reproduction script
  python3 trace_run.py -m pytest tests/test_x.py -q -x     # trace existing tests
Options (before the command): --root DIR (default: cwd)  --max-chars N (default 4000)
                              --timeout S (default 85)    --json OUT (keep the raw trace)
Output: exit code, the traceback tail, SUSPECTS (repo functions ranked: failure stack first, then
Ochiai over failing vs passing tests when both exist, then runtime distance from the failure),
and runtime calls into methods, which the precomputed code graph usually lacks.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))


def _tracelens_path() -> str:
    """Directory that contains the tracelens package (bundled next to this file, or repo src/)."""
    for cand in (HERE, os.path.join(HERE, "..", "..", "..", "src"), os.environ.get("TRACELENS_PATH", "")):
        if cand and os.path.isfile(os.path.join(cand, "tracelens", "tracer.py")):
            return os.path.abspath(cand)
    sys.exit("trace_run: tracelens package not found next to this script")


def _parse(argv):
    opts = {"root": os.getcwd(), "max_chars": 4000, "timeout": 85, "json": None}
    i = 0
    while i < len(argv) and argv[i] in ("--root", "--max-chars", "--timeout", "--json"):
        k = argv[i][2:].replace("-", "_")
        opts[k] = argv[i + 1] if k in ("root", "json") else int(argv[i + 1])
        i += 2
    cmd = argv[i:]
    if cmd[:1] == ["--"]:
        cmd = cmd[1:]
    if not cmd:
        sys.exit(__doc__)
    return opts, cmd


def is_test_path(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    base = parts[-1]
    return (base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py"
            or any(p in ("tests", "test", "testing") for p in parts[:-1]))


def _enclosing(nodes_by_file, rel, line):
    best = None
    for n in nodes_by_file.get(rel, ()):
        if n["firstlineno"] <= line and (best is None or n["firstlineno"] > best["firstlineno"]):
            best = n
    return best


def summarize(d: dict, stderr_tail: str, max_chars: int) -> str:
    nodes = {n["i"]: n for n in d.get("nodes", [])}
    by_file = defaultdict(list)
    for n in nodes.values():
        if n["qualname"] != "<module>":
            by_file[n["relpath"]].append(n)
    keep = lambda n: n["qualname"] != "<module>" and not is_test_path(n["relpath"])  # noqa: E731

    tests = d.get("tests") or {}
    failed = [t for t, v in tests.items() if v.get("outcome") in ("failed", "error")]
    passed = [t for t, v in tests.items() if v.get("outcome") == "passed"]
    frames = list(d.get("script_fail_frames") or [])
    for t in failed:
        frames += tests[t].get("fail_frames", [])

    # failure-stack nodes, innermost first
    stack = []
    for rel, line, _ in reversed(frames):
        n = _enclosing(by_file, rel, line)
        if n is not None and keep(n) and n["i"] not in stack:
            stack.append(n["i"])

    # runtime adjacency (undirected) and BFS distance from the failure stack
    adj = defaultdict(set)
    for e in d.get("edges", []):
        if e.get("caller") is not None:
            adj[e["caller"]].add(e["callee"])
            adj[e["callee"]].add(e["caller"])
    dist = {}
    if stack:
        dq = deque(stack)
        dist = {s: 0 for s in stack}
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    dq.append(w)

    ochiai = {}
    if failed and passed:
        fcnt, pcnt = defaultdict(int), defaultdict(int)
        for t in failed:
            for i in set(tests[t].get("nodes", [])):
                fcnt[i] += 1
        for t in passed:
            for i in set(tests[t].get("nodes", [])):
                pcnt[i] += 1
        for i in fcnt:
            ochiai[i] = fcnt[i] / math.sqrt(len(failed) * (fcnt[i] + pcnt[i]))

    cand = [n for n in nodes.values() if keep(n)]
    pos = {s: k for k, s in enumerate(stack)}

    def key(n):
        i = n["i"]
        return (pos.get(i, 10 ** 6), -ochiai.get(i, 0.0), dist.get(i, 10 ** 6), i)

    cand.sort(key=key)
    out = [f"TRACELENS exit_code={d.get('exit_code')} repo_functions_run={len(cand)} "
           f"tests_failed={len(failed)} tests_passed={len(passed)}"]
    if stderr_tail.strip():
        out.append("TRACEBACK / STDERR (tail):")
        out += ["  " + ln for ln in stderr_tail.strip().splitlines()[-14:]]
    if failed:
        out.append("FAILED TESTS: " + ", ".join(failed[:6]))
    if not cand:
        out.append("No repository function executed. Check the repro imports the package from /workspace.")
        return "\n".join(out)
    out.append("SUSPECTS (likely edit locations first; file:line function [why]):")
    for n in cand[:25]:
        why = []
        if n["i"] in pos:
            why.append("on failure stack")
        if n["i"] in ochiai:
            why.append(f"ochiai={ochiai[n['i']]:.2f}")
        if n["i"] in dist and n["i"] not in pos:
            why.append(f"{dist[n['i']]} call(s) from failure")
        why.append(f"calls={n.get('calls', 1)}")
        out.append(f"  {n['relpath']}:{n['firstlineno']} {n['qualname']} [{', '.join(why)}]")
    top = {n["i"] for n in cand[:25]}
    meth = []
    for e in d.get("edges", []):
        c, t = e.get("caller"), e.get("callee")
        if c is None or c not in nodes or t not in nodes:
            continue
        if "." in nodes[t]["qualname"] and keep(nodes[c]) and keep(nodes[t]) and (c in top or t in top):
            meth.append(e)
    if meth:
        out.append("RUNTIME CALLS INTO METHODS (the code graph usually lacks these):")
        for e in meth[:20]:
            a, b = nodes[e["caller"]], nodes[e["callee"]]
            via = " via framework" if e.get("kind") == "lifted" else ""
            out.append(f"  {a['qualname']} -> {b['qualname']} ({b['relpath']}:{b['firstlineno']}){via}")
    out.append("Next: read the top suspects with read_file, then fix the code (not the tests).")
    text = "\n".join(out)
    return text if len(text) <= max_chars else text[: max_chars - 16] + "\n...[truncated]"


def main(argv=None) -> int:
    opts, cmd = _parse(list(sys.argv[1:] if argv is None else argv))
    root = os.path.realpath(opts["root"])
    tmp = opts["json"] or os.path.join(tempfile.gettempdir(), f"tracelens_{os.getpid()}.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = _tracelens_path() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    full = [sys.executable, "-m", "tracelens.run", "--root", root, "--out", tmp, "--", *cmd]
    try:
        p = subprocess.run(full, cwd=root, env=env, capture_output=True, text=True, timeout=opts["timeout"])
        tail = (p.stderr or "")[-2500:]
        if not (p.stderr or "").strip() and p.returncode:
            tail = (p.stdout or "")[-2500:]
        tail = tail.replace(root + os.sep, "").replace(root + "/", "")
    except subprocess.TimeoutExpired:
        print(f"TRACELENS: command exceeded {opts['timeout']} s (possible hang = a finding); "
              "trace a smaller repro or a single test (-x -k NAME).")
        return 1
    if not os.path.exists(tmp):
        print("TRACELENS: no trace written. Output tail:\n" + tail)
        return 1
    with open(tmp, encoding="utf-8") as fh:
        d = json.load(fh)
    print(summarize(d, tail, opts["max_chars"]))
    if not opts["json"]:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
