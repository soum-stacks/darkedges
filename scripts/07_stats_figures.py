#!/usr/bin/env python3
"""T2.4 (HEL): statistics + figures F1-F4 + a hypotheses checklist (facts only, no prose for the paper).

Reads : results/dark_by_task.csv, rankings.csv, gold_in_trace.csv, linkpred.csv (whichever exist)
Writes: results/stats.json, results/hypotheses_summary.md, paper/figures/F1..F4 (.png and .pdf)
Run   : python scripts/07_stats_figures.py [--boot 10000] [--min-tasks 5]
Repos with fewer than --min-tasks tasks (httpx) are pooled in 'all' but get no per-repo claims.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkedges import stats as S  # noqa: E402
from darkedges.dataio import repo_root  # noqa: E402

REPO_ORDER = ["fastapi", "rich", "requests"]
METHODS = ["ochiai", "stack_dist", "static_k1", "static_k2", "static_k3", "bm25"]
KS = [5, 10, 20, 50]


def short(r):
    return str(r).split("/")[-1].lower()


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def savefig(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=S.B_DEFAULT)
    ap.add_argument("--min-tasks", type=int, default=5)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo_root()
    res, figs = root / "results", root / "paper" / "figures"
    B = a.boot

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: dict = {"boot": B, "min_tasks": a.min_tasks}
    md: list[str] = ["# T2 hypotheses checklist (auto-generated facts; interpretation is yours)\n"]

    # ------------------------------------------------------------ H1 dark fraction
    p = res / "dark_by_task.csv"
    repos_ok = []
    if p.exists() and p.stat().st_size:
        d = pd.read_csv(p)
        d["repo"] = d["repo"].map(short)
        counts = d.dropna(subset=["dark_fraction"]).groupby("repo").size()
        repos_ok = [r for r in REPO_ORDER if counts.get(r, 0) >= a.min_tasks]
        h1 = {"per_repo": {}, "per_repo_direct": {}, "contrasts": {}, "n_tasks_by_repo": counts.to_dict()}
        for r in repos_ok:
            h1["per_repo"][r] = S.cluster_bootstrap_mean(d[d.repo == r]["dark_fraction"], B)
            h1["per_repo_direct"][r] = S.cluster_bootstrap_mean(d[d.repo == r]["dark_fraction_direct"], B)
        h1["all"] = S.cluster_bootstrap_mean(d["dark_fraction"], B)
        for x, y in [("fastapi", "rich"), ("rich", "requests"), ("fastapi", "requests")]:
            if x in repos_ok and y in repos_ok:
                h1["contrasts"][f"{x}-{y}"] = S.cluster_bootstrap_group_diff(
                    d[d.repo == x]["dark_fraction"], d[d.repo == y]["dark_fraction"], B)
        if len(repos_ok) >= 2:
            order = sorted(repos_ok, key=lambda r: -h1["per_repo"][r]["mean"])
            h1["observed_order_high_to_low"] = order
            h1["H1_ordering_matches"] = bool(order[0] == "fastapi" and order[-1] == "requests")
        out["H1"] = h1
        md.append("## H1 dark-edge fraction (task-macro mean, headline = direct+lifted, runtime phase, non-test)\n")
        md.append("| repo | tasks | dark fraction [95% CI] | direct-only [95% CI] |\n|---|---|---|---|")
        for r in repos_ok:
            c, cd = h1["per_repo"][r], h1["per_repo_direct"][r]
            md.append(f"| {r} | {c['n']} | {fmt(c['mean'])} [{fmt(c['lo'])}, {fmt(c['hi'])}] | "
                      f"{fmt(cd['mean'])} [{fmt(cd['lo'])}, {fmt(cd['hi'])}] |")
        for k, v in h1["contrasts"].items():
            md.append(f"\n- {k}: diff {fmt(v['diff'])} [{fmt(v['lo'])}, {fmt(v['hi'])}]")
        if "observed_order_high_to_low" in h1:
            md.append(f"\nObserved order (high to low): {' > '.join(h1['observed_order_high_to_low'])}; "
                      f"H1 ordering (fastapi highest, requests lowest) matches: {h1['H1_ordering_matches']}\n")

        fig, ax = plt.subplots(figsize=(5, 3.4))
        xs = np.arange(len(repos_ok))
        for off, key, lab in [(-0.18, "per_repo", "direct + lifted"), (0.18, "per_repo_direct", "direct only")]:
            m = [h1[key][r]["mean"] for r in repos_ok]
            lo = [h1[key][r]["mean"] - h1[key][r]["lo"] for r in repos_ok]
            hi = [h1[key][r]["hi"] - h1[key][r]["mean"] for r in repos_ok]
            ax.bar(xs + off, m, 0.34, yerr=[lo, hi], capsize=3, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r}\n(n={h1['per_repo'][r]['n']})" for r in repos_ok])
        ax.set_ylabel("dark-edge fraction")
        ax.legend(frameon=False)
        savefig(fig, figs / "F1_dark_fraction")
        plt.close(fig)

    # ------------------------------------------------------------ H2 rankings
    p = res / "rankings.csv"
    if p.exists() and p.stat().st_size:
        r_ = pd.read_csv(p)
        r_["repo"] = r_["repo"].map(short)
        h2 = {"recall": {}, "contrasts": {}}
        for K in KS:
            for m in METHODS:
                v = r_[(r_.method == m) & (r_.K == K)]["hit"]
                h2["recall"][f"{m}@{K}"] = S.cluster_bootstrap_mean(v, B)
        piv = {K: r_[r_.K == K].pivot(index="instance_id", columns="method", values="hit") for K in KS}
        pivb = {K: r_[r_.K == K].pivot(index="instance_id", columns="method", values="hit_bin") for K in KS}
        pairs = [("ochiai", "static_k2"), ("stack_dist", "static_k2"), ("ochiai", "bm25"),
                 ("stack_dist", "bm25"), ("ochiai", "stack_dist"), ("bm25", "static_k2")]
        for K in KS:
            for x, y in pairs:
                if x in piv[K] and y in piv[K]:
                    c = S.cluster_bootstrap_diff(piv[K][x], piv[K][y], B)
                    c.update({"mcnemar": S.mcnemar_exact(pivb[K][x], pivb[K][y])})
                    c["meets_10pt"] = bool(c["diff"] is not None and c["diff"] >= 0.10)
                    c["ci_excludes_0"] = bool(c["lo"] is not None and (c["lo"] > 0 or c["hi"] < 0))
                    h2["contrasts"][f"{x}-{y}@{K}"] = c
        h2["by_repo_K20"] = {
            rr: {m: S.cluster_bootstrap_mean(r_[(r_.repo == rr) & (r_.method == m) & (r_.K == 20)]["hit"], B)
                 for m in METHODS} for rr in REPO_ORDER if (r_.repo == rr).any()}
        out["H2"] = h2
        md.append("## H2 recall@K (task-macro mean of tie-averaged hit)\n")
        md.append("| method | " + " | ".join(f"K={K}" for K in KS) + " |\n|---|" + "---|" * len(KS))
        for m in METHODS:
            md.append(f"| {m} | " + " | ".join(
                f"{fmt(h2['recall'][f'{m}@{K}']['mean'])} [{fmt(h2['recall'][f'{m}@{K}']['lo'])}, "
                f"{fmt(h2['recall'][f'{m}@{K}']['hi'])}]" for K in KS) + " |")
        md.append("\nPaired contrasts at K=20 (diff in recall, 95% CI, exact McNemar on hit_bin):\n")
        for x, y in pairs:
            c = h2["contrasts"].get(f"{x}-{y}@20")
            if c:
                md.append(f"- {x} - {y}: {fmt(c['diff'])} [{fmt(c['lo'])}, {fmt(c['hi'])}], n={c['n']}, "
                          f"McNemar p={fmt(c['mcnemar']['p'])} (only-{x} {c['mcnemar']['a_only']}, "
                          f"only-{y} {c['mcnemar']['b_only']}); meets >=10 pts: {c['meets_10pt']}")
        n_tasks = int(r_[(r_.method == "ochiai") & (r_.K == 20)].shape[0])
        out["H2"]["n_tasks"] = n_tasks

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for m in METHODS:
            ys = [h2["recall"][f"{m}@{K}"]["mean"] for K in KS]
            lo = [h2["recall"][f"{m}@{K}"]["lo"] for K in KS]
            hi = [h2["recall"][f"{m}@{K}"]["hi"] for K in KS]
            ax.plot(KS, ys, marker="o", label=m)
            ax.fill_between(KS, lo, hi, alpha=0.12)
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels(KS)
        ax.set_xlabel("K")
        ax.set_ylabel("recall of gold node @K")
        ax.legend(frameon=False, fontsize=8)
        savefig(fig, figs / "F2_recall_at_k")
        plt.close(fig)

    # ------------------------------------------------------------ gold in trace + Spearman
    p = res / "gold_in_trace.csv"
    if p.exists() and p.stat().st_size:
        g = pd.read_csv(p)
        g["repo"] = g["repo"].map(short)
        gt = {"overall": {}, "by_repo": {}}
        for col in ["in_trace", "in_trace_diff", "dark_only", "gold_in_static_k3_of_fail"]:
            gt["overall"][col] = S.cluster_bootstrap_mean(g[col], B)
            for rr in REPO_ORDER:
                if (g.repo == rr).sum() >= a.min_tasks:
                    gt["by_repo"].setdefault(rr, {})[col] = S.cluster_bootstrap_mean(g[g.repo == rr][col], B)
        out["gold_in_trace"] = gt
        md.append("\n## Gold edit location vs traces (fraction of tasks)\n")
        md.append("| scope | in_trace | in_trace_diff | dark_only | within static 3 hops of failing frame |\n|---|---|---|---|---|")
        for scope, dd in [("all", gt["overall"])] + list(gt["by_repo"].items()):
            md.append(f"| {scope} | " + " | ".join(fmt(dd[c]["mean"]) for c in
                      ["in_trace", "in_trace_diff", "dark_only", "gold_in_static_k3_of_fail"]) + " |")
        md.append(f"\nn tasks (gold in graph): {len(g)}")

        if p.exists() and (res / "dark_by_task.csv").exists():
            d2 = pd.read_csv(res / "dark_by_task.csv")[["instance_id", "dark_fraction"]]
            m = g.merge(d2, on="instance_id")
            sp = {"dark_fraction_vs_dark_only_all_tasks": S.spearman(m["dark_fraction"], m["dark_only"]),
                  "dark_fraction_vs_dark_only_in_trace_tasks":
                      S.spearman(m[m.in_trace == 1]["dark_fraction"], m[m.in_trace == 1]["dark_only"])}
            out["spearman"] = sp
            md.append("\nSpearman (task level): " + "; ".join(
                f"{k}: rho={fmt(v['rho'])}, p={fmt(v['p'])}, n={v['n']}" for k, v in sp.items()))

        fig, ax = plt.subplots(figsize=(5, 3.4))
        cols = [("in_trace", "in trace"), ("in_trace_diff", "in failing, not passing"), ("dark_only", "dark only")]
        scopes = [("all", gt["overall"])] + list(gt["by_repo"].items())
        xs = np.arange(len(scopes))
        for j, (c, lab) in enumerate(cols):
            ax.bar(xs + (j - 1) * 0.26, [dd[c]["mean"] for _, dd in scopes], 0.26, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([s for s, _ in scopes])
        ax.set_ylabel("fraction of tasks")
        ax.legend(frameon=False, fontsize=8)
        savefig(fig, figs / "F3_gold_in_trace")
        plt.close(fig)

    # ------------------------------------------------------------ H3 link prediction
    p = res / "linkpred.csv"
    if p.exists() and p.stat().st_size:
        lp = pd.read_csv(p)
        h3 = {"auc": {}, "emb_minus_name_auc": {}, "emb_beats_name_by_0.02": {}}
        for rr, grp in lp.groupby("heldout_repo"):
            h3["auc"][rr] = dict(zip(grp.scorer, grp.auc))
            sc = h3["auc"][rr]
            if "emb_cos" in sc and "name_jaccard" in sc:
                diff = sc["emb_cos"] - sc["name_jaccard"]
                h3["emb_minus_name_auc"][rr] = diff
                h3["emb_beats_name_by_0.02"][rr] = bool(diff > 0.02)
        out["H3"] = h3
        md.append("\n## H3 link prediction (AUC, leave-one-repo-out)\n")
        scorers = list(dict.fromkeys(lp.scorer))
        md.append("| held-out | " + " | ".join(scorers) + " |\n|---|" + "---|" * len(scorers))
        for rr, sc in h3["auc"].items():
            md.append(f"| {rr} | " + " | ".join(fmt(sc.get(s)) for s in scorers) + " |")
        if h3["emb_minus_name_auc"]:
            md.append("\nemb_cos - name_jaccard AUC: " + ", ".join(
                f"{k}: {fmt(v)}" for k, v in h3["emb_minus_name_auc"].items()))
            md.append(f"\nEmbeddings beat the name-token baseline by >0.02 AUC in "
                      f"{sum(h3['emb_beats_name_by_0.02'].values())}/{len(h3['emb_beats_name_by_0.02'])} held-out repos.")

        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        reps = list(h3["auc"])
        xs = np.arange(len(scorers))
        w = 0.8 / max(1, len(reps))
        for j, rr in enumerate(reps):
            ax.bar(xs + (j - (len(reps) - 1) / 2) * w, [h3["auc"][rr].get(s, np.nan) for s in scorers], w, label=rr)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(scorers, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUC (held-out repo)")
        ax.legend(frameon=False, fontsize=8)
        savefig(fig, figs / "F4_linkpred_auc")
        plt.close(fig)

    (res / "stats.json").write_text(json.dumps(out, indent=2, default=float))
    (res / "hypotheses_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {res / 'stats.json'}, {res / 'hypotheses_summary.md'}, figures in {figs}")


if __name__ == "__main__":
    main()
