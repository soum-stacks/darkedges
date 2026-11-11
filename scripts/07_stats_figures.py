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
        series = [("dark_fraction", "headline (direct+lifted)"), ("dark_fraction_direct", "direct only"),
                  ("dark_fraction_excl_same_class", "excl. same-class pairs"),
                  ("dark_fraction_strict", "excl. same-class pairs and dunder callees")]
        series = [(c, l) for c, l in series if c in d.columns]
        h1 = {"series": {c: {} for c, _ in series}, "contrasts": {}, "n_tasks_by_repo": counts.to_dict()}
        for c, _ in series:
            for r in repos_ok:
                h1["series"][c][r] = S.cluster_bootstrap_mean(d[d.repo == r][c], B)
            h1["series"][c]["all"] = S.cluster_bootstrap_mean(d[c], B)
        for c in [c for c, _ in series if c in ("dark_fraction", "dark_fraction_excl_same_class", "dark_fraction_strict")]:
            for x, y in [("fastapi", "rich"), ("rich", "requests"), ("fastapi", "requests")]:
                if x in repos_ok and y in repos_ok:
                    h1["contrasts"][f"{c}:{x}-{y}"] = S.cluster_bootstrap_group_diff(
                        d[d.repo == x][c], d[d.repo == y][c], B)
        for c, _ in series:
            order = sorted(repos_ok, key=lambda r: -h1["series"][c][r]["mean"])
            h1["series"][c]["order_high_to_low"] = order
            h1["series"][c]["H1_ordering_matches"] = bool(order and order[0] == "fastapi" and order[-1] == "requests")
        # pooled composition and near-miss counts (task-pair level; duplicates across tasks counted)
        sums = {k: int(d[k].sum()) for k in d.columns if k.startswith("n_") and d[k].dtype.kind in "iu"}
        h1["pooled_counts"] = sums
        out["H1"] = h1
        md.append("## H1 dark-edge fraction (task-macro mean; runtime phase, non-test, kind direct|lifted)\n")
        md.append("| repo | tasks | " + " | ".join(l for _, l in series) + " |\n|---|---|" + "---|" * len(series))
        for r in repos_ok:
            md.append(f"| {r} | {h1['series'][series[0][0]][r]['n']} | " + " | ".join(
                f"{fmt(h1['series'][c][r]['mean'])} [{fmt(h1['series'][c][r]['lo'])}, {fmt(h1['series'][c][r]['hi'])}]"
                for c, _ in series) + " |")
        md.append("")
        for k, v in h1["contrasts"].items():
            md.append(f"- {k}: diff {fmt(v['diff'])} [{fmt(v['lo'])}, {fmt(v['hi'])}]")
        for c, l in series:
            md.append(f"- order by {l}: {' > '.join(h1['series'][c]['order_high_to_low'])}; "
                      f"H1 ordering (fastapi highest, requests lowest) matches: {h1['series'][c]['H1_ordering_matches']}")
        md.append("\nNote: requests has only a handful of tasks; its intervals are unreliable.")
        if "n_dark" in sums and sums.get("n_dark"):
            nd = sums["n_dark"]
            md.append("\n### Composition of dark pairs (pooled over tasks; rich dominates the pair counts)\n")
            md.append(f"- pairs {sums.get('n_pairs')}, dark {nd}")
            for k, lab in [("n_dark_same_class", "caller and callee in the same class"),
                           ("n_dark_dunder_callee", "callee is a dunder method"),
                           ("n_dark_len2", "static path length 2 (class nodes allowed as intermediates)"),
                           ("n_dark_len2_noclass", "static path length 2 without class nodes"),
                           ("n_dark_len3", "static path length 3 (class nodes allowed)"),
                           ("n_dark_len3_noclass", "static path length 3 without class nodes")]:
                if k in sums:
                    md.append(f"- {lab}: {sums[k]} ({sums[k] / nd:.1%} of dark pairs)")

        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        xs = np.arange(len(repos_ok))
        shown = [(c, l) for c, l in series if c != "dark_fraction_direct"]
        w = 0.8 / max(1, len(shown))
        for j, (c, l) in enumerate(shown):
            m = [h1["series"][c][r]["mean"] for r in repos_ok]
            lo = [h1["series"][c][r]["mean"] - h1["series"][c][r]["lo"] for r in repos_ok]
            hi = [h1["series"][c][r]["hi"] - h1["series"][c][r]["mean"] for r in repos_ok]
            ax.bar(xs + (j - (len(shown) - 1) / 2) * w, m, w, yerr=[lo, hi], capsize=2, label=l)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r}\n(n={h1['series']['dark_fraction'][r]['n']})" for r in repos_ok])
        ax.set_ylabel("dark-edge fraction")
        ax.legend(frameon=False, fontsize=7)
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

    # ------------------------------------------------------------ H3 link prediction (all variants)
    lp_files = sorted(res.glob("linkpred*.csv"))
    h3_all = {}
    for p in lp_files:
        if not p.stat().st_size:
            continue
        variant = p.stem.replace("linkpred", "").strip("_") or "default"
        lp = pd.read_csv(p)
        h3 = {"auc": {}, "hits10": {}, "emb_minus_name_auc": {}, "emb_beats_name_by_0.02": {}}
        for rr, grp in lp.groupby("heldout_repo"):
            h3["auc"][rr] = dict(zip(grp.scorer, grp.auc))
            h3["hits10"][rr] = dict(zip(grp.scorer, grp.hits10))
            sc = h3["auc"][rr]
            if "emb_cos" in sc and "name_jaccard" in sc:
                diff = sc["emb_cos"] - sc["name_jaccard"]
                h3["emb_minus_name_auc"][rr] = diff
                h3["emb_beats_name_by_0.02"][rr] = bool(diff > 0.02)
        h3["n_pos"] = {rr: int(grp.n_pos.iloc[0]) for rr, grp in lp.groupby("heldout_repo")}
        h3_all[variant] = h3
        md.append(f"\n## H3 link prediction, variant '{variant}' (AUC, leave-one-repo-out; pooled over held-out tasks, no intervals)\n")
        scorers = list(dict.fromkeys(lp.scorer))
        md.append("| held-out (n_pos) | " + " | ".join(scorers) + " |\n|---|" + "---|" * len(scorers))
        for rr, sc in h3["auc"].items():
            md.append(f"| {rr} ({h3['n_pos'][rr]}) | " + " | ".join(fmt(sc.get(s_)) for s_ in scorers) + " |")
        md.append("\nHits@10:\n")
        md.append("| held-out | " + " | ".join(scorers) + " |\n|---|" + "---|" * len(scorers))
        for rr, sc in h3["hits10"].items():
            md.append(f"| {rr} | " + " | ".join(fmt(sc.get(s_)) for s_ in scorers) + " |")
        if h3["emb_minus_name_auc"]:
            md.append("\nemb_cos - name_jaccard AUC: " + ", ".join(f"{k}: {fmt(v)}" for k, v in h3["emb_minus_name_auc"].items())
                      + f"; embeddings beat the name-token baseline by >0.02 AUC in "
                        f"{sum(h3['emb_beats_name_by_0.02'].values())}/{len(h3['emb_beats_name_by_0.02'])} held-out repos.")
        if "logreg" in scorers and "logreg_no_emb" in scorers:
            md.append("logreg - logreg_no_emb AUC (value of embeddings on top of the other scorers): " + ", ".join(
                f"{rr}: {fmt(sc['logreg'] - sc['logreg_no_emb'])}" for rr, sc in h3["auc"].items()))
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        reps = list(h3["auc"])
        xs = np.arange(len(scorers))
        w = 0.8 / max(1, len(reps))
        for j, rr in enumerate(reps):
            ax.bar(xs + (j - (len(reps) - 1) / 2) * w, [h3["auc"][rr].get(s_, np.nan) for s_ in scorers], w, label=rr)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(scorers, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUC (held-out repo)")
        ax.set_title(variant, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        savefig(fig, figs / ("F4_linkpred_auc" if variant == "default" else f"F4_linkpred_auc_{variant}"))
        plt.close(fig)
    if h3_all:
        out["H3"] = h3_all

    (res / "stats.json").write_text(json.dumps(out, indent=2, default=float))
    (res / "hypotheses_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {res / 'stats.json'}, {res / 'hypotheses_summary.md'}, figures in {figs}")


if __name__ == "__main__":
    main()
