#!/usr/bin/env python3
"""Exploratory (post hoc, logged in docs/t3_choices.md): can a cheap STATIC pass recover the dark edges?

Repair rules, applied to the released graph using only node ids and node text (no execution, no traces):
  R1 self-calls    : inside a method of class C, `self.m(...)` / `cls.m(...)` -> C.m if that node exists,
                     else the same method on a base class of C (bases parsed from the class header,
                     resolved within the same module or by a unique class short name), up to 3 levels.
  R2 + attributes  : R1 plus `self.m` / `cls.m` attribute loads without a call (properties, bound methods).
  R3 + constructors: R2 plus `X(...)` where X resolves (same module, else unique short name) to a class
                     node with an `X.__init__` node -> edge to `X.__init__`.
Measured on the T2 labelled runtime pairs (headline filter: runtime phase, kind direct|lifted, no test code,
unique (caller, callee) pair per task) and on the static k-hop localisation baseline (issue-symbol seeds,
undirected, any edge type) re-run on the repaired graph. The traces are used only to SCORE the repair.

Reads : data/slim, data/traces, data/extras/gold_nodes.jsonl, results/edges_labeled.jsonl
Writes: results/repair_by_task.csv, results/repair_rankings.csv, results/repair_summary.json,
        results/repair_summary.md (aggregates), paper/figures/F7_repair.png/.pdf
Usage : python scripts/13_graph_repair.py [--task-set strict] [--boot 10000] [--root REPO] [--limit N]
"""
import argparse
import ast
import csv
import json
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darkedges.dataio import (GraphCache, bfs_levels, find_graph, load_tasks, load_trace, read_jsonl,  # noqa: E402
                              repo_root, repo_short, select_tasks)
from darkedges.rankings import KS, _seed_for, build_name_index, candidate_ok, rank_eval, seeds_from_issue  # noqa: E402
from darkedges.stats import cluster_bootstrap_diff, cluster_bootstrap_mean, mcnemar_exact  # noqa: E402

VARIANTS = ("R1", "R2", "R3")


# ----------------------------------------------------------------- static repair
def _parse(text: str):
    src = textwrap.dedent(text or "")
    for cand in (src, src + "\n    pass\n"):
        try:
            return ast.parse(cand)
        except SyntaxError:
            continue
    return None


def _class_bases(graph, cls_id: str, short_classes: dict) -> list[str]:
    tree = _parse(graph.text.get(cls_id, ""))
    if tree is None or not tree.body or not isinstance(tree.body[0], ast.ClassDef):
        return []
    module = cls_id.rsplit(".", 1)[0]
    out = []
    for b in tree.body[0].bases:
        name = b.attr if isinstance(b, ast.Attribute) else (b.id if isinstance(b, ast.Name) else None)
        if not name:
            continue
        if f"{module}.{name}" in graph.class_nodes:
            out.append(f"{module}.{name}")
        elif len(short_classes.get(name, ())) == 1:
            out.append(short_classes[name][0])
    return out


def _resolve_method(graph, cls_id, m, bases_of, depth=0):
    if cls_id is None or depth > 3:
        return None
    if f"{cls_id}.{m}" in graph.node_set:
        return f"{cls_id}.{m}"
    for b in bases_of(cls_id):
        hit = _resolve_method(graph, b, m, bases_of, depth + 1)
        if hit:
            return hit
    return None


def repair_edges(graph) -> dict:
    """Return {variant: set((caller, callee))} of edges absent from the graph's calls edges."""
    short_classes = defaultdict(list)
    for c in graph.class_nodes:
        short_classes[c.rsplit(".", 1)[-1]].append(c)
    base_cache: dict = {}

    def bases_of(c):
        if c not in base_cache:
            base_cache[c] = _class_bases(graph, c, short_classes)
        return base_cache[c]

    add = {v: set() for v in VARIANTS}
    for n in graph.ids:
        if n in graph.class_nodes:
            continue
        tree = _parse(graph.text.get(n, ""))
        if tree is None:
            continue
        cls = graph.class_of(n)
        module = n.rsplit(".", 1)[0] if cls is None else cls.rsplit(".", 1)[0]
        called_attr_nodes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and f.value.id in ("self", "cls") and cls is not None):
                    called_attr_nodes.add(id(f))
                    t = _resolve_method(graph, cls, f.attr, bases_of)
                    if t and t != n:
                        add["R1"].add((n, t))
                elif isinstance(f, ast.Name):
                    tc = f"{module}.{f.id}" if f"{module}.{f.id}" in graph.class_nodes else (
                        short_classes[f.id][0] if len(short_classes.get(f.id, ())) == 1 else None)
                    if tc and f"{tc}.__init__" in graph.node_set and f"{tc}.__init__" != n:
                        add["R3"].add((n, f"{tc}.__init__"))
        if cls is not None:
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and id(node) not in called_attr_nodes
                        and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")
                        and isinstance(node.ctx, ast.Load)):
                    t = _resolve_method(graph, cls, node.attr, bases_of)
                    if t and t != n:
                        add["R2"].add((n, t))
    add["R2"] |= add["R1"]
    add["R3"] |= add["R2"]
    for v in VARIANTS:
        add[v] = {(u, w) for u, w in add[v] if not graph.has_call(u, w)}
    return add


def callee_type(graph, caller, callee):
    cc = graph.class_of(callee)
    if cc is None:
        return "function"
    return "method_same_class" if cc == graph.class_of(caller) else "method_other_class"


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-set", default="strict", choices=["strict", "f2p"])
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo_root()
    slim, traces, res = root / "data" / "slim", root / "data" / "traces", root / "results"
    gold = {r["instance_id"]: set(r["gold_node_ids"]) for r in read_jsonl(root / "data" / "extras" / "gold_nodes.jsonl")}

    pairs = defaultdict(dict)   # iid -> {(u, v): dark}
    for line in open(res / "edges_labeled.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["phase"] != "runtime" or r.get("test_code") or r["kind"] not in ("direct", "lifted"):
            continue
        key = (r["caller_id"], r["callee_id"])
        pairs[r["instance_id"]][key] = pairs[r["instance_id"]].get(key, False) or (not r["static_edge"])

    chosen = select_tasks(load_tasks(slim), traces, a.task_set)
    if a.limit:
        chosen = chosen[: a.limit]
    cache = GraphCache(maxsize=3)
    rep_cache: dict = {}
    by_task, rank_rows, pooled = [], [], defaultdict(lambda: defaultdict(int))
    added_stats = {}
    t0 = time.time()
    for k, (task, pre) in enumerate(chosen, 1):
        iid, repo = task["instance_id"], repo_short(task["repo"])
        gp = find_graph(slim, task)
        if gp is None:
            continue
        graph = cache.get(gp)
        if gp not in rep_cache:
            rep_cache.clear()
            add = repair_edges(graph)
            rep_adj = {}
            for v in VARIANTS:
                adj = defaultdict(set, {u: set(ws) for u, ws in graph.adj_all.items()})
                for u, w in add[v]:
                    adj[u].add(w)
                    adj[w].add(u)
                rep_adj[v] = adj
            rep_cache[gp] = (add, rep_adj, build_name_index(graph))
            added_stats[gp.name] = {"repo": repo, "calls_edges": sum(len(x) for x in graph.calls_out.values()),
                                    **{f"added_{v}": len(add[v]) for v in VARIANTS}}
        add, rep_adj, nidx = rep_cache[gp]

        # (1) dark-edge recovery on this task's runtime pairs
        P = pairs.get(iid, {})
        row = {"instance_id": iid, "repo": repo, "n_pairs": len(P), "n_dark": sum(P.values())}
        executed = {u for u, _ in P} | {w for _, w in P}
        for v in VARIANTS:
            rec = sum(1 for e, d in P.items() if d and e in add[v])
            row[f"recovered_{v}"] = rec
            row[f"dark_fraction_{v}"] = (row["n_dark"] - rec) / len(P) if P else None
            cand = [e for e in add[v] if e[0] in executed and e[1] in executed]
            row[f"added_between_executed_{v}"] = len(cand)
            row[f"added_confirmed_{v}"] = sum(1 for e in cand if e in P)
        row["dark_fraction_orig"] = row["n_dark"] / len(P) if P else None
        for e, d in P.items():
            ct = callee_type(graph, *e)
            pooled[ct]["pairs"] += 1
            pooled[ct]["dark"] += int(d)
            for v in VARIANTS:
                pooled[ct][f"rec_{v}"] += int(d and e in add[v])
        by_task.append(row)

        # (2) static k-hop localisation before / after repair (non-oracle baseline)
        g = {x for x in gold.get(iid, set()) if x in graph.node_set}
        if g:
            seeds = seeds_from_issue(task.get("problem_statement", ""), graph, nidx)
            for kk in (1, 2, 3):
                variants = {"orig": graph.adj_all, **{v: rep_adj[v] for v in VARIANTS}}
                for name, adj in variants.items():
                    if seeds:
                        dist = bfs_levels(adj, seeds, kk)
                        scores = {n: -float(d) for n, d in dist.items() if candidate_ok(n)}
                    else:
                        scores = {}
                    r = rank_eval(scores, g, KS, a.shuffles, _seed_for(iid, f"static_k{kk}"))
                    for K in KS:
                        rank_rows.append({"instance_id": iid, "repo": repo, "method": f"static_k{kk}", "graph": name,
                                          "K": K, "hit": r["hit"][K], "hit_bin": int(r["hit"][K] >= 0.5),
                                          "n_candidates": r["n_candidates"]})
        if k % 10 == 0:
            print(f"  {k}/{len(chosen)} tasks, {time.time() - t0:.0f}s", flush=True)

    def write_csv(p, rows):
        if rows:
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    write_csv(res / "repair_by_task.csv", by_task)
    write_csv(res / "repair_rankings.csv", rank_rows)

    # ---- aggregates
    S = {"task_set": a.task_set, "n_tasks": len(by_task), "graphs": added_stats, "dark_fraction": {},
         "pooled_by_callee_type": {k: dict(v) for k, v in pooled.items()}, "precision_proxy": {}, "localisation": {}}
    for repo in sorted({r["repo"] for r in by_task}):
        sub = [r for r in by_task if r["repo"] == repo and r["n_pairs"]]
        S["dark_fraction"][repo] = {"n_tasks": len(sub), "orig": cluster_bootstrap_mean(
            [r["dark_fraction_orig"] for r in sub], B=a.boot)}
        for v in VARIANTS:
            S["dark_fraction"][repo][v] = cluster_bootstrap_mean([r[f"dark_fraction_{v}"] for r in sub], B=a.boot)
            S["dark_fraction"][repo][f"share_of_dark_recovered_{v}"] = (
                sum(r[f"recovered_{v}"] for r in sub) / max(1, sum(r["n_dark"] for r in sub)))
    for v in VARIANTS:
        cand = sum(r[f"added_between_executed_{v}"] for r in by_task)
        conf = sum(r[f"added_confirmed_{v}"] for r in by_task)
        S["precision_proxy"][v] = {"added_edges_between_executed_functions": cand, "observed_at_runtime": conf,
                                   "rate": conf / cand if cand else None}
    idx = defaultdict(dict)
    for r in rank_rows:
        idx[(r["method"], r["K"], r["graph"])][r["instance_id"]] = r
    for m in ("static_k1", "static_k2", "static_k3"):
        for K in KS:
            base = idx[(m, K, "orig")]
            ids = sorted(base)
            ent = {"n_tasks": len(ids), "orig": cluster_bootstrap_mean([base[i]["hit"] for i in ids], B=a.boot)}
            for v in VARIANTS:
                rep = idx[(m, K, v)]
                ent[v] = cluster_bootstrap_mean([rep[i]["hit"] for i in ids], B=a.boot)
                ent[f"{v}_minus_orig"] = cluster_bootstrap_diff([rep[i]["hit"] for i in ids],
                                                                [base[i]["hit"] for i in ids], B=a.boot)
                ent[f"{v}_mcnemar"] = mcnemar_exact([rep[i]["hit_bin"] for i in ids], [base[i]["hit_bin"] for i in ids])
                ent[f"{v}_median_candidates"] = float(np.median([rep[i]["n_candidates"] for i in ids])) if ids else None
            ent["orig_median_candidates"] = float(np.median([base[i]["n_candidates"] for i in ids])) if ids else None
            S["localisation"][f"{m}@{K}"] = ent
    (res / "repair_summary.json").write_text(json.dumps(S, indent=1))

    f = lambda m: "n/a" if m.get("mean") is None else f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"  # noqa: E731
    fd = lambda m: "n/a" if m.get("diff") is None else f"{m['diff']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]"  # noqa: E731
    L = ["# Static graph repair (exploratory, post hoc) - facts only", "",
         f"Task set {a.task_set}; {len(by_task)} tasks. Unit: unique runtime (caller, callee) pair per task; task-macro"
         " mean with 95% task-clustered CI.", "",
         "## Dark fraction before / after repair", "| Repo | tasks | original | R1 self-calls | R2 +attributes | R3 +constructors |"
         " share of dark pairs recovered (R1/R2/R3) |", "|---|---|---|---|---|---|---|"]
    for repo, d in S["dark_fraction"].items():
        L.append(f"| {repo} | {d['n_tasks']} | {f(d['orig'])} | {f(d['R1'])} | {f(d['R2'])} | {f(d['R3'])} | "
                 + "/".join(f"{d[f'share_of_dark_recovered_{v}']:.3f}" for v in VARIANTS) + " |")
    L += ["", "## Pooled by callee type (pairs, dark, recovered R1/R2/R3)"]
    for ct, d in sorted(S["pooled_by_callee_type"].items()):
        L.append(f"- {ct}: pairs {d.get('pairs', 0)}, dark {d.get('dark', 0)}, recovered "
                 + "/".join(str(d.get(f"rec_{v}", 0)) for v in VARIANTS))
    L += ["", "## Precision proxy (added edges whose two endpoints both ran; share observed as runtime pairs)"]
    for v, d in S["precision_proxy"].items():
        L.append(f"- {v}: {d['observed_at_runtime']}/{d['added_edges_between_executed_functions']}"
                 f" = {d['rate'] if d['rate'] is None else round(d['rate'], 3)}")
    L += ["", "## Static k-hop localisation recall@K (issue-symbol seeds), original vs repaired graph",
          "| method@K | tasks | original | R3 | R3 - original [CI] | McNemar R3-only/orig-only, p | median candidates orig/R3 |",
          "|---|---|---|---|---|---|---|"]
    for key, e in S["localisation"].items():
        mc = e["R3_mcnemar"]
        L.append(f"| {key} | {e['n_tasks']} | {f(e['orig'])} | {f(e['R3'])} | {fd(e['R3_minus_orig'])} | "
                 f"{mc['a_only']}/{mc['b_only']}, p={mc['p']:.3g} | {e['orig_median_candidates']}/{e['R3_median_candidates']} |")
    L += ["", "Added edges per graph: see repair_summary.json['graphs']."]
    (res / "repair_summary.md").write_text("\n".join(L) + "\n")
    figure(S, root / "paper" / "figures" / "F7_repair")
    print("\n".join(L))


def figure(S, out_base: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_base.parent.mkdir(parents=True, exist_ok=True)
    repos = [r for r in ("fastapi", "rich", "requests") if r in S["dark_fraction"]]
    steps = [("orig", "Released graph"), ("R1", "+ self-calls"), ("R2", "+ attribute refs"), ("R3", "+ constructors")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for j, r in enumerate(repos):
        ys = [S["dark_fraction"][r][s]["mean"] for s, _ in steps]
        los = [S["dark_fraction"][r][s]["lo"] for s, _ in steps]
        his = [S["dark_fraction"][r][s]["hi"] for s, _ in steps]
        xs = np.arange(len(steps)) + (j - 1) * 0.08
        ax.plot(xs, ys, "-o", color=colors[j], lw=2, ms=7, mec="white", mew=2, label=f"{r} (n={S['dark_fraction'][r]['n_tasks']})")
        ax.vlines(xs, los, his, color=colors[j], lw=1.2, alpha=0.7)
    ax.set_xticks(range(len(steps)), [s[1] for s in steps], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("dark fraction of runtime pairs", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
