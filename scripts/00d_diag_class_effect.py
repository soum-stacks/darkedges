#!/usr/bin/env python3
"""Read-only diagnostic (HEL login node, ~10 s): how much do class nodes and same-class calls shape the dark edges?
Prints counts only (symbol ids are not printed). Usage: python scripts/00d_diag_class_effect.py [--root .]"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkedges.causes import is_class_text  # noqa: E402
from darkedges.dataio import GraphCache, bfs_levels, find_graph, load_tasks, repo_root  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
a = ap.parse_args()
root = Path(a.root).resolve() if a.root else repo_root()
slim, res = root / "data" / "slim", root / "results"
tasks = {t["instance_id"]: t for t in load_tasks(slim)}
cache = GraphCache(maxsize=3)
by_task = defaultdict(list)
for line in open(res / "edges_labeled.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["phase"] == "runtime" and not r["test_code"] and r["kind"] in ("direct", "lifted"):
        by_task[r["instance_id"]].append(r)


def class_of(g, nid, memo):
    if nid not in memo:
        parts, c = nid.split("."), None
        for i in range(len(parts) - 1, 0, -1):
            p = ".".join(parts[:i])
            if p in g.node_set and is_class_text(g.text.get(p, "")):
                c = p
                break
        memo[nid] = c
    return memo[nid]


tot = Counter()
per_repo = defaultdict(list)
for iid, rows in sorted(by_task.items()):
    g = cache.get(find_graph(slim, tasks[iid]))
    memo, cls = {}, {n for n in g.ids if is_class_text(g.text[n])}
    # calls graph with class nodes removed (they cannot act as intermediates)
    adj = {u: {v for v in vs if v not in cls} for u, vs in g.calls_out.items() if u not in cls}
    pairs = {}
    for r in rows:
        pairs.setdefault((r["caller_id"], r["callee_id"]), not r["static_edge"])
    n = nd = n_nsc = nd_nsc = 0
    for (u, v), dark in pairs.items():
        same = class_of(g, u, memo) is not None and class_of(g, u, memo) == class_of(g, v, memo)
        n += 1; nd += dark
        if not same:
            n_nsc += 1; nd_nsc += dark
        if dark:
            tot["dark_pairs"] += 1
            tot["same_class"] += same
            tot["callee_dunder"] += v.rsplit(".", 1)[-1].startswith("__")
            d = bfs_levels(adj, [u], 3).get(v)
            tot["noclass_len%s" % d] += 1
    repo = tasks[iid]["repo"].split("/")[-1]
    per_repo[repo].append((nd / n if n else None, nd_nsc / n_nsc if n_nsc else None))
tot_calls = Counter()
for t in list({id(find_graph(slim, tasks[i])): tasks[i] for i in by_task}.values())[:1]:
    g = cache.get(find_graph(slim, t))
    cls = {n for n in g.ids if is_class_text(g.text[n])}
    tot_calls["calls edges from class nodes"] = sum(len(g.calls_out.get(c, ())) for c in cls)
    tot_calls["calls edges from function nodes"] = sum(len(v) for u, v in g.calls_out.items() if u not in cls)
print("dark pairs (task-pair level, duplicates across tasks counted):", tot["dark_pairs"])
print("  caller and callee in the same class:", tot["same_class"])
print("  callee is a dunder/private-dunder method:", tot["callee_dunder"])
print("  static path length once class nodes are removed:",
      {k.replace("noclass_len", ""): v for k, v in tot.items() if k.startswith("noclass_len")}, "(None = no path within 3)")
print("example graph:", dict(tot_calls))
print("task-macro dark fraction: all pairs vs excluding same-class pairs")
for repo, v in sorted(per_repo.items()):
    x = [p[0] for p in v if p[0] is not None]; y = [p[1] for p in v if p[1] is not None]
    print(f"  {repo:9s} n={len(v):3d}  all={sum(x) / len(x):.3f}  excl_same_class={sum(y) / len(y):.3f}")
