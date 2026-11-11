#!/usr/bin/env python3
"""T2 closing check + figure F5 (HEL, ~20 s). Needs results/edges_labeled.jsonl and results/callee_split.csv.

(a) static graph: calls edges by (source kind -> target kind), kind in {class, method, function}, one graph per repo
(b) per task: share of runtime pairs whose callee is a method vs the task's dark fraction (Pearson, Spearman)
(c) figure paper/figures/F5_callee_type.png|pdf from results/callee_split.csv
Usage: python scripts/10_method_share.py [--root .]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkedges import stats as S  # noqa: E402
from darkedges.dataio import GraphCache, find_graph, load_tasks, repo_root  # noqa: E402


def kind_of(g, n):
    if n in g.class_nodes:
        return "class"
    return "method" if g.class_of(n) is not None else "function"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--min-tasks", type=int, default=5)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo_root()
    slim, res, figs = root / "data" / "slim", root / "results", root / "paper" / "figures"
    tasks = {t["instance_id"]: t for t in load_tasks(slim)}
    by_task = defaultdict(list)
    for line in open(res / "edges_labeled.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["phase"] == "runtime" and not r["test_code"] and r["kind"] in ("direct", "lifted"):
            by_task[r["instance_id"]].append(r)
    cache = GraphCache(maxsize=3)

    # (a) edge kinds in one graph per repo
    print("(a) static `calls` edges by source kind -> target kind (one graph per repo)")
    seen = set()
    for iid in sorted(by_task):
        repo = tasks[iid]["repo"].split("/")[-1].lower()
        if repo in seen:
            continue
        seen.add(repo)
        g = cache.get(find_graph(slim, tasks[iid]))
        c = Counter()
        for u, vs in g.calls_out.items():
            for v in vs:
                c[(kind_of(g, u), kind_of(g, v))] += 1
        tot = sum(c.values())
        print(f"  {repo:9s} total={tot}  " + "  ".join(f"{k[0]}->{k[1]}:{v}" for k, v in sorted(c.items())))
        into_m = sum(v for k, v in c.items() if k[1] == "method")
        print(f"            calls edges whose target is a method: {into_m} ({into_m / tot:.1%})")

    # (b) per-task relation
    rows = []
    for iid, rr in by_task.items():
        g = cache.get(find_graph(slim, tasks[iid]))
        pairs = {}
        for r in rr:
            pairs.setdefault((r["caller_id"], r["callee_id"]), not r["static_edge"])
        if not pairs:
            continue
        ms = np.mean([g.class_of(v) is not None for (_, v) in pairs])
        rows.append((tasks[iid]["repo"].split("/")[-1].lower(), ms, np.mean(list(pairs.values()))))
    print("\n(b) per task: share of pairs with a method callee vs dark fraction")
    for label, sel in [("all", rows)] + [(r, [x for x in rows if x[0] == r]) for r in ("fastapi", "rich", "requests")]:
        if len(sel) < a.min_tasks and label != "all":
            continue
        x, y = [s[1] for s in sel], [s[2] for s in sel]
        sp = S.spearman(x, y)
        pr = float(np.corrcoef(x, y)[0, 1]) if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")
        print(f"  {label:9s} n={len(sel):3d}  Spearman rho={sp['rho']}  p={sp['p']}  Pearson r={pr:.3f}  (r^2={pr * pr:.3f})")

    # (c) figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    cs = pd.read_csv(res / "callee_split.csv")
    cats = [("callee: module-level function", "module-level function"), ("callee: method, same class", "method, same class"),
            ("callee: method, other class", "method, other class")]
    repos = [r for r in ("fastapi", "rich", "requests") if (cs.repo == r).any() and
             cs[(cs.repo == r) & (cs.category == "all pairs")].n_tasks.iloc[0] >= a.min_tasks]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    bottom = np.zeros(len(repos))
    for cat, lab in cats:
        sh = np.array([cs[(cs.repo == r) & (cs.category == cat)].share_of_pairs.iloc[0] for r in repos])
        ax[0].bar(repos, sh, bottom=bottom, label=lab)
        bottom += sh
    ax[0].set_ylabel("share of runtime pairs")
    ax[0].set_title("callee type", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7)
    w = 0.26
    for j, (cat, lab) in enumerate(cats):
        sub = [cs[(cs.repo == r) & (cs.category == cat)].iloc[0] for r in repos]
        m = [s.dark_fraction_macro for s in sub]
        ax[1].bar(np.arange(len(repos)) + (j - 1) * w, m, w, yerr=[[a_ - b_ for a_, b_ in zip(m, [s.lo for s in sub])],
                  [b_ - a_ for a_, b_ in zip(m, [s.hi for s in sub])]], capsize=2, label=lab)
    ax[1].set_xticks(np.arange(len(repos)))
    ax[1].set_xticklabels(repos)
    ax[1].set_ylabel("dark fraction (task-macro)")
    ax[1].set_title("dark fraction within each callee type", fontsize=9)
    figs.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figs / "F5_callee_type.png", dpi=200)
    fig.savefig(figs / "F5_callee_type.pdf")
    print(f"\n(c) wrote {figs / 'F5_callee_type.png'}")


if __name__ == "__main__":
    main()