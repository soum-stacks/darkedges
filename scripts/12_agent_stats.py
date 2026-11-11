#!/usr/bin/env python3
"""T3.3 analysis (HEL): H4 and the agent-check table from results/agent_check.jsonl.

H4 (protocol.yaml, frozen): TraceLens raises the gold-file-touched rate for E4B over static graph tools,
paired by task and seed. Primary test: exact McNemar on (task, seed) pairs, tracelens vs static_graph_tools.
Effect size: per-task difference of seed-averaged rates with a task-clustered bootstrap CI (10,000 resamples).
Secondary (labelled): resolved; the other two contrasts; tool use; exploratory splits (repo, dark_only gold).

Reads : results/agent_check.jsonl (from scripts/08_agent_check on Kaggle), results/gold_in_trace.csv (optional)
Writes: results/agent_check_summary.json, results/agent_check_summary.md (aggregates only),
        paper/figures/F6_agent_check.png/.pdf
Usage : python scripts/12_agent_stats.py [--root REPO] [--boot 10000]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darkedges.dataio import repo_root, repo_short  # noqa: E402
from darkedges.stats import cluster_bootstrap_diff, cluster_bootstrap_mean, mcnemar_exact  # noqa: E402

CONDS = ["no_graph_tools", "static_graph_tools", "tracelens"]
LABEL = {"no_graph_tools": "No graph tools", "static_graph_tools": "Static graph tools", "tracelens": "TraceLens"}
COLOR = {"no_graph_tools": "#eb6834", "static_graph_tools": "#2a78d6", "tracelens": "#1baf7a"}
CONTRASTS = [("tracelens", "static_graph_tools", "H4 (pre-registered)"),
             ("tracelens", "no_graph_tools", "secondary"),
             ("static_graph_tools", "no_graph_tools", "secondary")]
METRICS = ["gold_file_touched", "resolved"]


def load(path: Path):
    last = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            last[(d["instance_id"], d["condition"], int(d["seed"]))] = d
    ok = {k: v for k, v in last.items() if not (v.get("error") or "").startswith("EXC:")}
    bad = sorted(k for k in last if k not in ok)
    return ok, bad


def per_task(rows: dict, cond: str, metric: str, tasks: list, seeds: list) -> np.ndarray:
    out = []
    for t in tasks:
        vals = [float(rows[(t, cond, s)][metric]) for s in seeds if (t, cond, s) in rows
                and rows[(t, cond, s)].get(metric) is not None]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out, dtype=float)


def paired_bins(rows, a, b, metric, tasks, seeds):
    xa, xb = [], []
    for t in tasks:
        for s in seeds:
            if (t, a, s) in rows and (t, b, s) in rows:
                xa.append(int(bool(rows[(t, a, s)][metric])))
                xb.append(int(bool(rows[(t, b, s)][metric])))
    return xa, xb


def fmt(m):
    return "n/a" if m.get("mean") is None else f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def fmtd(m):
    return "n/a" if m.get("diff") is None else f"{m['diff']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]"


def figure(summary: dict, out_base: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, metric, title in zip(axes, METRICS, ["Patch touches a gold file", "Resolved (tests pass)"]):
        for y, c in enumerate(CONDS):
            m = summary["by_condition"][c][metric]
            if m["mean"] is None:
                continue
            ax.plot([m["lo"], m["hi"]], [y, y], color=COLOR[c], lw=2, solid_capstyle="round")
            ax.plot([m["mean"]], [y], "o", ms=8, color=COLOR[c], mec="white", mew=2)
            ax.text(m["hi"] + 0.02, y, f"{m['mean']:.2f}", va="center", fontsize=8, color="#444444")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(len(CONDS) - 0.5, -0.5)
        ax.set_xlabel("rate (task mean of seed means), 95% task-clustered CI", fontsize=7)
    axes[0].set_yticks(range(len(CONDS)), [LABEL[c] for c in CONDS], fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo_root()
    res = root / "results"
    src = res / "agent_check.jsonl"
    if not src.exists():
        sys.exit(f"missing {src}: unzip agent_check_outputs.zip from Kaggle into results/agent_check_raw and "
                 f"copy agent_check.jsonl to results/")
    rows, bad = load(src)
    tasks = sorted({k[0] for k in rows})
    seeds = sorted({k[2] for k in rows})
    repo_of = {k[0]: repo_short(v["repo"]) for k, v in rows.items()}
    S = {"n_tasks": len(tasks), "seeds": seeds, "n_runs": len(rows), "missing_or_failed_runs": [list(b) for b in bad],
         "by_condition": {}, "contrasts": [], "tool_use": {}, "exploratory": {}}

    for c in CONDS:
        S["by_condition"][c] = {}
        for m in METRICS + ["tool_calls", "seconds", "patch_chars"]:
            S["by_condition"][c][m] = cluster_bootstrap_mean(per_task(rows, c, m, tasks, seeds), B=a.boot)
        S["by_condition"][c]["n_runs"] = sum(1 for k in rows if k[1] == c)
        pcs = [rows[k]["patch_chars"] for k in rows if k[1] == c and rows[k].get("patch_chars") is not None]
        S["by_condition"][c]["empty_patch_rate"] = float(np.mean([x == 0 for x in pcs])) if pcs else None

    for x, y, tag in CONTRASTS:
        for m in METRICS:
            d = cluster_bootstrap_diff(per_task(rows, x, m, tasks, seeds), per_task(rows, y, m, tasks, seeds), B=a.boot)
            mc = mcnemar_exact(*paired_bins(rows, x, y, m, tasks, seeds))
            S["contrasts"].append({"a": x, "b": y, "metric": m, "tag": tag if m == "gold_file_touched" else "secondary",
                                   "paired_diff": d, "mcnemar": mc})

    tl = [v for k, v in rows.items() if k[1] == "tracelens"]
    if tl:
        used = [v for v in tl if v.get("trace_ok", 0) > 0]
        S["tool_use"] = {
            "tracelens_runs": len(tl), "runs_with_tracer_output": len(used),
            "tracer_use_rate": len(used) / len(tl),
            "gold_file_rate_when_tracer_used": float(np.mean([v["gold_file_touched"] for v in used])) if used else None,
            "gold_file_rate_when_not_used": float(np.mean([v["gold_file_touched"] for v in tl if v.get("trace_ok", 0) == 0]))
            if len(used) < len(tl) else None,
            "mean_graph_mentions": {c: (float(np.mean(g)) if (g := [v["graph_mentions"] for k, v in rows.items()
                                        if k[1] == c and v.get("graph_mentions") is not None]) else None)
                                    for c in CONDS},
        }

    # exploratory: per repo (fastapi, rich only) and by dark_only gold (from T2)
    for r in ("fastapi", "rich"):
        sub = [t for t in tasks if repo_of[t] == r]
        if len(sub) >= 5:
            S["exploratory"][f"repo_{r}"] = {
                "n_tasks": len(sub),
                **{f"{c}_gold_file": cluster_bootstrap_mean(per_task(rows, c, "gold_file_touched", sub, seeds), B=a.boot)
                   for c in CONDS}}
    git = res / "gold_in_trace.csv"
    if git.exists():
        with open(git, newline="", encoding="utf-8") as fh:
            dark = {r["instance_id"]: int(str(r.get("dark_only") or "0").strip().lower() in ("1", "1.0", "true"))
                    for r in csv.DictReader(fh)}
        for flag in (1, 0):
            sub = [t for t in tasks if dark.get(t) == flag]
            if sub:
                S["exploratory"][f"dark_only_gold={flag}"] = {
                    "n_tasks": len(sub),
                    "diff_tracelens_minus_static": cluster_bootstrap_diff(
                        per_task(rows, "tracelens", "gold_file_touched", sub, seeds),
                        per_task(rows, "static_graph_tools", "gold_file_touched", sub, seeds), B=a.boot)}

    (res / "agent_check_summary.json").write_text(json.dumps(S, indent=1))
    L = [f"# Agent check (T3.3) - facts only\n",
         f"E4B; tasks {len(tasks)}; seeds {seeds}; runs {len(rows)}; failed/missing runs {len(bad)} (listed in json).",
         "Unit: per-task mean over seeds; 95% task-clustered bootstrap CI. McNemar on (task, seed) pairs.\n",
         "| Condition | gold file touched | resolved | tool calls | seconds | empty patch |", "|---|---|---|---|---|---|"]
    for c in CONDS:
        b = S["by_condition"][c]
        L.append(f"| {LABEL[c]} | {fmt(b['gold_file_touched'])} | {fmt(b['resolved'])} | "
                 f"{b['tool_calls']['mean'] if b['tool_calls']['mean'] is None else round(b['tool_calls']['mean'], 1)} | "
                 f"{b['seconds']['mean'] if b['seconds']['mean'] is None else round(b['seconds']['mean'], 1)} | "
                 f"{b['empty_patch_rate'] if b['empty_patch_rate'] is None else round(b['empty_patch_rate'], 3)} |")
    L += ["", "| Contrast | metric | tag | paired diff [95% CI] | McNemar a_only/b_only, p |", "|---|---|---|---|---|"]
    for x in S["contrasts"]:
        mc = x["mcnemar"]
        L.append(f"| {LABEL[x['a']]} - {LABEL[x['b']]} | {x['metric']} | {x['tag']} | {fmtd(x['paired_diff'])} | "
                 f"{mc['a_only']}/{mc['b_only']}, p={mc['p']:.3g} |")
    h4 = next(x for x in S["contrasts"] if x["tag"].startswith("H4"))
    sup = h4["paired_diff"]["diff"] is not None and h4["paired_diff"]["lo"] > 0 and h4["mcnemar"]["p"] < 0.05
    L += ["", f"H4 supported at alpha 0.05 (CI excludes 0 and McNemar p < 0.05, before Holm): {sup}",
          "Holm across H1-H4 must be applied with the H1-H3 p-values from results/stats.json.", "",
          "Tool use: " + json.dumps(S["tool_use"]), "", "Exploratory (post hoc, not pre-registered): "
          + json.dumps(S["exploratory"], default=str)]
    (res / "agent_check_summary.md").write_text("\n".join(L) + "\n")
    figs = root / "paper" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    figure(S, figs / "F6_agent_check")
    print("\n".join(L))
    print(f"\nwrote {res / 'agent_check_summary.json'}, {res / 'agent_check_summary.md'}, {figs / 'F6_agent_check.png'}")


if __name__ == "__main__":
    main()
