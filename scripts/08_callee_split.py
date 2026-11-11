#!/usr/bin/env python3
"""Extra 1 (HEL): where are the dark edges? Dark fraction by callee type, per repo.

Tests the idea that dark edges concentrate in method calls across classes (calls on objects) rather than in
calls to module-level functions. Reads results/edges_labeled.jsonl (step 04) and the graphs.
Writes results/callee_split.csv and prints a table. Categories overlap (they are different cuts), see CATS.
Usage: python scripts/08_callee_split.py [--root .] [--boot 2000]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkedges import stats as S  # noqa: E402
from darkedges.causes import is_dunder  # noqa: E402
from darkedges.dataio import GraphCache, find_graph, load_tasks, repo_root  # noqa: E402
from darkedges.linkpred import module_of  # noqa: E402

CATS = {
    "all pairs": lambda p: True,
    "callee: module-level function": lambda p: not p["callee_method"],
    "callee: method, same class": lambda p: p["callee_method"] and p["same_class"],
    "callee: method, other class": lambda p: p["callee_method"] and not p["same_class"],
    "callee: method, other class, non-dunder": lambda p: p["callee_method"] and not p["same_class"] and not p["dunder"],
    "callee: dunder method": lambda p: p["dunder"],
    "same module": lambda p: p["same_module"],
    "cross module": lambda p: not p["same_module"],
}


def categorize(rows, g):
    """rows: labeled edge rows of ONE task (headline set). Returns one dict per unique (caller, callee) pair."""
    pairs = {}
    for r in rows:
        pairs.setdefault((r["caller_id"], r["callee_id"]), not r["static_edge"])
    out = []
    for (u, v), dark in pairs.items():
        cu, cv = g.class_of(u), g.class_of(v)
        out.append({"dark": dark, "callee_method": cv is not None, "same_class": cu is not None and cu == cv,
                    "dunder": is_dunder(v.rsplit(".", 1)[-1]), "same_module": module_of(u) == module_of(v)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min-tasks", type=int, default=5)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo_root()
    slim, res = root / "data" / "slim", root / "results"
    tasks = {t["instance_id"]: t for t in load_tasks(slim)}
    by_task = defaultdict(list)
    for line in open(res / "edges_labeled.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["phase"] == "runtime" and not r["test_code"] and r["kind"] in ("direct", "lifted"):
            by_task[r["instance_id"]].append(r)

    cache = GraphCache(maxsize=3)
    per = defaultdict(lambda: defaultdict(list))      # repo -> cat -> [(n, dark) per task]
    for iid, rows in sorted(by_task.items()):
        g = cache.get(find_graph(slim, tasks[iid]))
        pairs = categorize(rows, g)
        repo = tasks[iid]["repo"].split("/")[-1].lower()
        for name, f in CATS.items():
            sel = [p for p in pairs if f(p)]
            per[repo][name].append((len(sel), sum(p["dark"] for p in sel)))

    out = []
    for repo, cats in sorted(per.items()):
        total_pairs = sum(n for n, _ in cats["all pairs"])
        for name, tl in cats.items():
            fr = [d / n for n, d in tl if n]
            ci = S.cluster_bootstrap_mean(fr, a.boot) if len(fr) >= 2 else {"mean": None, "lo": None, "hi": None}
            n_pairs, n_dark = sum(n for n, _ in tl), sum(d for _, d in tl)
            out.append({"repo": repo, "category": name, "n_tasks": len(fr), "n_pairs": n_pairs,
                        "share_of_pairs": n_pairs / total_pairs if total_pairs else None,
                        "dark_fraction_macro": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                        "dark_fraction_pooled": (n_dark / n_pairs) if n_pairs else None})
    res.mkdir(exist_ok=True)
    with open(res / "callee_split.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    f = lambda x: "  n/a" if x is None else f"{x:5.3f}"  # noqa: E731
    print("dark fraction (task-macro mean [95% CI]); n_pairs; share of the repo's pairs")
    for repo in sorted(per):
        n_t = len(per[repo]["all pairs"])
        if n_t < a.min_tasks:
            continue
        print(f"\n{repo} ({n_t} tasks)")
        for r in [x for x in out if x["repo"] == repo]:
            sh = f"{r['share_of_pairs']:.1%}" if r["share_of_pairs"] is not None else "n/a"
            print(f"  {r['category']:42s} {f(r['dark_fraction_macro'])} [{f(r['lo'])}, {f(r['hi'])}]  "
                  f"pairs={r['n_pairs']:6d} share={sh:>6s}  tasks={r['n_tasks']}")
    print(f"\nwrote {res / 'callee_split.csv'}")


if __name__ == "__main__":
    main()
