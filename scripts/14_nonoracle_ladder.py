#!/usr/bin/env python3
"""Exploratory (see docs/PRECOMMIT_ladder.md): a NON-ORACLE runtime localisation ladder.

How much of the oracle runtime advantage survives when only tests an agent could run BEFORE the fix are used?
Rungs: R0 BM25 | R1 coverage ceiling | R3 issue-retrieved existing tests + trace specificity | R4 RRF(BM25, R3) |
R4b BM25 restricted to the retrieved tests' functions | R5 oracle Ochiai.

Reads : data/slim (tasks, graphs), data/traces/<id>.pre.json, data/extras/gold_nodes.jsonl
Writes: results/ladder_rows.csv, results/ladder_summary.json, results/ladder_summary.md, paper/figures/F8_ladder.png/.pdf
Usage : python scripts/14_nonoracle_ladder.py [--k-tests 5] [--boot 10000] [--shuffles 100] [--root REPO] [--limit N]
"""
import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darkedges.compare import map_trace  # noqa: E402
from darkedges.dataio import (GraphCache, find_graph, load_tasks, load_trace, read_jsonl, repo_root,  # noqa: E402
                              repo_short, select_tasks)
from darkedges.rankings import (BM25, _seed_for, candidate_ok, eligible_map, mapped_ids_of_nodes,  # noqa: E402
                                ochiai_scores, rank_eval, tokenize)
from darkedges.stats import cluster_bootstrap_diff, cluster_bootstrap_mean, mcnemar_exact  # noqa: E402

KS = (1, 5, 10, 20, 50)
RUNGS = ["R0_bm25", "R3_tests_idf", "R4_hybrid_rrf", "R4b_bm25_in_tests", "R5_oracle_ochiai"]
LABEL = {"R0_bm25": "BM25 (issue text only)", "R3_tests_idf": "Issue-retrieved existing tests",
         "R4_hybrid_rrf": "Hybrid: BM25 + existing tests", "R4b_bm25_in_tests": "BM25 within tests' functions",
         "R5_oracle_ochiai": "Oracle: Ochiai with failing tests"}


# ----------------------------------------------------------------- leakage filter
def patch_test_info(test_patch: str):
    """Files that are NEW in the test patch, and test function names it adds or edits."""
    new_files = set()
    for m in re.finditer(r"^--- (\S+)\s*\n\+\+\+ b/(\S+)", test_patch or "", re.M):
        if m.group(1) == "/dev/null":
            new_files.add(m.group(2))
    names = set(re.findall(r"^\+\s*(?:async\s+)?def\s+(test\w*)", test_patch or "", re.M))
    names |= set(re.findall(r"^@@[^@]*@@.*?def\s+(test\w*)", test_patch or "", re.M))
    return new_files, names


def existing_passing_tests(trace: dict, test_patch: str) -> list[str]:
    new_files, names = patch_test_info(test_patch)
    out = []
    for nid, v in trace.get("tests", {}).items():
        if v.get("outcome") != "passed":
            continue
        f = nid.split("::")[0]
        fn = nid.split("::")[-1].split("[")[0]
        if f in new_files or fn in names:
            continue
        out.append(nid)
    return out


# ----------------------------------------------------------------- rungs
def test_documents(trace, mapping, graph, tests):
    """Token list per test: nodeid tokens + source of the test function when it maps to a graph node."""
    by_i = {n["i"]: n for n in trace["nodes"]}
    docs = {}
    for t in tests:
        toks = tokenize(t.replace("::", " ").replace("/", " ").replace(".py", " "))
        fn = t.split("::")[-1].split("[")[0]
        for i in trace["tests"][t].get("nodes", []):
            n = by_i.get(i)
            if n is not None and n["qualname"].split(".")[-1] == fn and mapping.get(i):
                toks += tokenize(graph.text.get(mapping[i], ""))
                break
        docs[t] = toks
    return docs


def rrf(*rankings, c=60):
    out = defaultdict(float)
    for sc in rankings:
        order = sorted(sc, key=lambda n: (-sc[n], n))
        for r, n in enumerate(order, 1):
            out[n] += 1.0 / (c + r)
    return dict(out)


def ladder_task(task, trace, graph, gold, bm25, k_tests, n_shuffles):
    iid, repo = task["instance_id"], repo_short(task["repo"])
    issue = tokenize(task.get("problem_statement", ""))
    mapping = map_trace(trace, graph)
    elig = eligible_map(trace, mapping)
    tests = existing_passing_tests(trace, task.get("test_patch", ""))
    exec_by_test = {t: mapped_ids_of_nodes(trace, mapping, trace["tests"][t].get("nodes", []), elig) for t in tests}

    scores = {"R0_bm25": bm25.scores(issue)}
    # R1 coverage ceiling
    covered = set().union(*exec_by_test.values()) if exec_by_test else set()
    # R2/R3: retrieve existing tests by the issue, weight functions by trace specificity
    n_f = defaultdict(int)
    for fs in exec_by_test.values():
        for f in fs:
            n_f[f] += 1
    N = len(tests)
    r3, cand = {}, set()
    if tests:
        tbm = BM25(test_documents(trace, mapping, graph, tests))
        sims = tbm.scores(issue)
        top = sorted(sims, key=lambda t: (-sims[t], t))[:k_tests]
        for t in top:
            for f in exec_by_test[t]:
                cand.add(f)
                r3[f] = r3.get(f, 0.0) + sims[t] * math.log(N / (1 + n_f[f]) + 1e-9 if N else 1.0)
        r3 = {f: s for f, s in r3.items() if s > 0} or {f: 1e-6 for f in cand}
    scores["R3_tests_idf"] = r3
    scores["R4_hybrid_rrf"] = rrf(scores["R0_bm25"], r3) if r3 else dict(scores["R0_bm25"])
    scores["R4b_bm25_in_tests"] = {f: s for f, s in scores["R0_bm25"].items() if f in cand}
    och, _ = ochiai_scores(trace, mapping)
    scores["R5_oracle_ochiai"] = och

    rows = []
    for m in RUNGS:
        r = rank_eval(scores[m], gold, KS, n_shuffles, _seed_for(iid, "ladder_" + m))
        for K in KS:
            rows.append({"instance_id": iid, "repo": repo, "rung": m, "K": K, "hit": r["hit"][K],
                         "hit_bin": int(r["hit"][K] >= 0.5), "n_candidates": r["n_candidates"]})
    info = {"instance_id": iid, "repo": repo, "n_existing_tests": N, "n_tests_total": len(trace.get("tests", {})),
            "gold_covered_by_existing": int(bool(gold & covered)), "gold_in_retrieved": int(bool(gold & cand)),
            "n_retrieved_candidates": len(cand), "retrieved_any": int(bool(cand))}
    return rows, info


# ----------------------------------------------------------------- main
def boot_gap(r0, r4, r5, B, seed=0):
    r0, r4, r5 = map(lambda x: np.asarray(x, float), (r0, r4, r5))
    n = len(r0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    den = r5[idx].mean(1) - r0[idx].mean(1)
    num = r4[idx].mean(1) - r0[idx].mean(1)
    g = np.where(np.abs(den) > 1e-9, num / np.where(np.abs(den) > 1e-9, den, 1), np.nan)
    g = g[~np.isnan(g)]
    point = (r4.mean() - r0.mean()) / (r5.mean() - r0.mean()) if abs(r5.mean() - r0.mean()) > 1e-9 else None
    return {"gap_closed": point, "lo": float(np.quantile(g, 0.025)) if len(g) else None,
            "hi": float(np.quantile(g, 0.975)) if len(g) else None}


def figure(S, out_base: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["R0_bm25", "R3_tests_idf", "R4b_bm25_in_tests", "R4_hybrid_rrf", "R5_oracle_ochiai"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for y, m in enumerate(order):
        e = S["pooled"][m]["20"]
        col = "#8a8a8a" if m == "R5_oracle_ochiai" else ("#2a78d6" if m.startswith("R4") else "#1baf7a")
        ax.barh(y, e["mean"], color=col, height=0.55, alpha=0.35 if m == "R5_oracle_ochiai" else 0.9,
                hatch="//" if m == "R5_oracle_ochiai" else None, edgecolor=col)
        ax.plot([e["lo"], e["hi"]], [y, y], color="#333333", lw=1.2)
        ax.text(e["hi"] + 0.015, y, f"{e['mean']:.2f}", va="center", fontsize=8, color="#333333")
    cov = S["coverage_ceiling"]["mean"]
    ax.axvline(cov, color="#eb6834", ls="--", lw=1.2)
    ax.text(cov, len(order) - 0.45, f"gold run by existing tests: {cov:.2f}", fontsize=7, color="#eb6834", ha="center")
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order], fontsize=8)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall@20 (task mean, 95% task-clustered CI)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-tests", type=int, default=5)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo_root()
    slim, traces, res = root / "data" / "slim", root / "data" / "traces", root / "results"
    res.mkdir(exist_ok=True)
    gold_all = {r["instance_id"]: set(r["gold_node_ids"]) for r in read_jsonl(root / "data" / "extras" / "gold_nodes.jsonl")}
    chosen = select_tasks(load_tasks(slim), traces, "strict")
    if a.limit:
        chosen = chosen[: a.limit]
    cache, bm_cache = GraphCache(maxsize=3), {}
    rows, infos, skipped = [], [], 0
    t0 = time.time()
    for k, (task, pre) in enumerate(chosen, 1):
        gp = find_graph(slim, task)
        if gp is None:
            skipped += 1
            continue
        graph = cache.get(gp)
        gold = {g for g in gold_all.get(task["instance_id"], set()) if g in graph.node_set}
        if not gold:
            skipped += 1
            continue
        if gp not in bm_cache:
            bm_cache.clear()
            bm_cache[gp] = BM25({n: tokenize(graph.text[n]) for n in graph.ids if candidate_ok(n)})
        r, info = ladder_task(task, load_trace(pre), graph, gold, bm_cache[gp], a.k_tests, a.shuffles)
        rows += r
        infos.append(info)
        if k % 10 == 0:
            print(f"  {k}/{len(chosen)} tasks, {time.time() - t0:.0f}s", flush=True)

    with open(res / "ladder_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    idx = defaultdict(dict)
    for r in rows:
        idx[(r["rung"], r["K"])][r["instance_id"]] = r
    ids = sorted({r["instance_id"] for r in rows})
    repo_of = {i["instance_id"]: i["repo"] for i in infos}
    S = {"n_tasks": len(ids), "skipped": skipped, "k_tests": a.k_tests,
         "existing_tests_total": sum(i["n_existing_tests"] for i in infos),
         "tests_total": sum(i["n_tests_total"] for i in infos),
         "tasks_with_no_existing_test": sum(1 for i in infos if i["n_existing_tests"] == 0),
         "tasks_with_retrieved_tests": sum(i["retrieved_any"] for i in infos),
         "coverage_ceiling": cluster_bootstrap_mean([i["gold_covered_by_existing"] for i in infos], B=a.boot),
         "gold_in_retrieved_candidates": cluster_bootstrap_mean([i["gold_in_retrieved"] for i in infos], B=a.boot),
         "pooled": {}, "by_repo": {}, "contrasts": {}}
    for m in RUNGS:
        S["pooled"][m] = {str(K): cluster_bootstrap_mean([idx[(m, K)][i]["hit"] for i in ids], B=a.boot) for K in KS}
    for repo in ("fastapi", "rich", "requests"):
        sub = [i for i in ids if repo_of[i] == repo]
        if sub:
            S["by_repo"][repo] = {"n": len(sub), **{m: cluster_bootstrap_mean([idx[(m, 20)][i]["hit"] for i in sub],
                                                                               B=a.boot)["mean"] for m in RUNGS}}
    for x, y in [("R4_hybrid_rrf", "R0_bm25"), ("R3_tests_idf", "R0_bm25"), ("R4b_bm25_in_tests", "R0_bm25"),
                 ("R5_oracle_ochiai", "R4_hybrid_rrf")]:
        for K in (5, 20):
            S["contrasts"][f"{x}-{y}@{K}"] = {
                **cluster_bootstrap_diff([idx[(x, K)][i]["hit"] for i in ids], [idx[(y, K)][i]["hit"] for i in ids], B=a.boot),
                "mcnemar": mcnemar_exact([idx[(x, K)][i]["hit_bin"] for i in ids], [idx[(y, K)][i]["hit_bin"] for i in ids])}
    S["gap_closed_at_20"] = boot_gap([idx[("R0_bm25", 20)][i]["hit"] for i in ids],
                                     [idx[("R4_hybrid_rrf", 20)][i]["hit"] for i in ids],
                                     [idx[("R5_oracle_ochiai", 20)][i]["hit"] for i in ids], a.boot)
    (res / "ladder_summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")

    f = lambda e: "n/a" if e.get("mean") is None else f"{e['mean']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"  # noqa: E731
    L = ["# Non-oracle runtime ladder (exploratory; docs/PRECOMMIT_ladder.md) - facts only", "",
         f"Tasks {S['n_tasks']} (skipped {skipped}); existing passing tests used {S['existing_tests_total']} of "
         f"{S['tests_total']} traced; tasks with no existing test {S['tasks_with_no_existing_test']}; "
         f"tasks where the issue retrieved >=1 test {S['tasks_with_retrieved_tests']}; k = {a.k_tests}.",
         f"Coverage ceiling (gold executed by any existing test): {f(S['coverage_ceiling'])}",
         f"Gold inside the retrieved tests' functions: {f(S['gold_in_retrieved_candidates'])}", "",
         "| Rung | R@1 | R@5 | R@10 | R@20 | R@50 |", "|---|---|---|---|---|---|"]
    for m in RUNGS:
        L.append(f"| {LABEL[m]} | " + " | ".join(f"{S['pooled'][m][str(K)]['mean']:.3f}" for K in KS) + " |")
    L.append(f"\nR@20 with CI: " + "; ".join(f"{LABEL[m]} {f(S['pooled'][m]['20'])}" for m in RUNGS))
    L += ["", "| Contrast | diff [95% CI] | McNemar a_only/b_only, p |", "|---|---|---|"]
    for key, c in S["contrasts"].items():
        L.append(f"| {key} | {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | "
                 f"{c['mcnemar']['a_only']}/{c['mcnemar']['b_only']}, p={c['mcnemar']['p']:.3g} |")
    g = S["gap_closed_at_20"]
    L += ["", f"Gap closed at K=20, (R4 - R0) / (R5 - R0): {g['gap_closed'] if g['gap_closed'] is None else round(g['gap_closed'], 3)} "
              f"[{g['lo'] if g['lo'] is None else round(g['lo'], 3)}, {g['hi'] if g['hi'] is None else round(g['hi'], 3)}]",
          "", "R@20 by repo: " + json.dumps(S["by_repo"])]
    (res / "ladder_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    figure(S, root / "paper" / "figures" / "F8_ladder")
    print("\n".join(L))


if __name__ == "__main__":
    main()
