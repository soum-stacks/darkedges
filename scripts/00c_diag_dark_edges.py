#!/usr/bin/env python3
"""Diagnostics on the real data (HEL, login node is fine, ~10 s). Run after step 04.

Prints: how the graph's own `name` field relates to the node id; how many tasks get issue-text
seeds for the static baseline; what sits between caller and callee for 'near-miss' dark edges
(static path length 2); and the cause breakdown. Only symbol ids are printed, no source text.
Usage: python scripts/00c_diag_dark_edges.py [--root .] [--sample 2000]
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkedges.causes import is_class_text  # noqa: E402
from darkedges.dataio import GraphCache, find_graph, load_tasks, repo_root, select_tasks  # noqa: E402
from darkedges.rankings import build_name_index, seeds_from_issue  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--sample", type=int, default=2000)
a = ap.parse_args()
root = Path(a.root).resolve() if a.root else repo_root()
slim, traces, res = root / "data" / "slim", root / "data" / "traces", root / "results"
tasks = {t["instance_id"]: t for t in load_tasks(slim)}
cache = GraphCache(maxsize=3)

# ---- 1. graph `name` field vs id
chosen = select_tasks(list(tasks.values()), traces, "strict")
t0 = chosen[0][0]
g = cache.get(find_graph(slim, t0))
c = Counter()
for n in g.ids:
    raw = g.raw_name[n]
    c["raw==short" if raw == n.rsplit(".", 1)[-1] else "raw==id" if raw == n else "missing" if raw is None else "other"] += 1
print("1. graph name field vs id:", dict(c))
for n in g.ids[:4]:
    print("   id=%r  raw name=%r" % (n, g.raw_name[n]))

# ---- 2. seeds for the static baseline
no_seed, n_seed = 0, []
for t, _ in chosen:
    gg = cache.get(find_graph(slim, t))
    s = seeds_from_issue(t.get("problem_statement", ""), gg, build_name_index(gg))
    n_seed.append(len(s))
    no_seed += not s
n_seed.sort()
print(f"2. tasks with no issue-text seeds: {no_seed}/{len(chosen)}; seeds per task min/median/max = "
      f"{n_seed[0]}/{n_seed[len(n_seed) // 2]}/{n_seed[-1]}")

# ---- 3. near-miss dark edges: what is the intermediate node?
rows = []
with open(res / "edges_labeled.jsonl", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if (r["phase"] == "runtime" and not r["test_code"] and not r["static_edge"]
                and r["kind"] in ("direct", "lifted")):
            rows.append(r)
print(f"3. dark runtime rows: {len(rows)}; causes: {dict(Counter(r['cause'] for r in rows))}")
print("   static_path_len:", dict(Counter(r["static_path_len"] for r in rows)))
near = [r for r in rows if r["static_path_len"] == 2]
random.seed(0)
random.shuffle(near)
inter_class = inter_any = 0
ex = []
for r in near[: a.sample]:
    gg = cache.get(find_graph(slim, tasks[r["instance_id"]]))
    mids = [m for m in gg.calls_out.get(r["caller_id"], ()) if r["callee_id"] in gg.calls_out.get(m, ())]
    if not mids:
        continue
    inter_any += 1
    if any(is_class_text(gg.text.get(m, "")) for m in mids):
        inter_class += 1
    if len(ex) < 8:
        ex.append((r["caller_id"], mids[0], r["callee_id"], is_class_text(gg.text.get(mids[0], ""))))
k = min(len(near), a.sample)
print(f"   path-2 sample {k}: intermediate found for {inter_any}; at least one intermediate is a CLASS node: {inter_class}")
for caller, mid, callee, isc in ex:
    print(f"   {caller} -> [{mid}{' (class)' if isc else ''}] -> {callee}")
