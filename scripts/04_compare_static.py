#!/usr/bin/env python3
"""T2.1 (HEL): label every runtime edge as static or dark.

Reads : data/slim/tasks.jsonl, data/slim/graphs/*.json, data/traces/<id>.pre.json
Writes: results/edges_labeled.jsonl
        results/mapping_coverage.csv
        results/dark_by_task.csv
        results/compare_log.txt   (tasks skipped and why)
Run   : python scripts/04_compare_static.py [--task-set strict|f2p|all] [--limit N]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkedges.compare import coverage_row, label_edges, map_trace, task_dark_summary  # noqa: E402
from darkedges.dataio import GraphCache, find_graph, load_tasks, load_trace, repo_root, select_tasks  # noqa: E402


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-set", default="strict", choices=["strict", "f2p", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    root = Path(a.root).resolve() if a.root else repo_root()
    slim, traces, res = root / "data" / "slim", root / "data" / "traces", root / "results"
    res.mkdir(exist_ok=True)

    tasks = load_tasks(slim)
    chosen = select_tasks(tasks, traces, a.task_set)
    if a.limit:
        chosen = chosen[: a.limit]
    print(f"task set '{a.task_set}': {len(chosen)} of {len(tasks)} tasks")

    cache = GraphCache(maxsize=4)
    cov_rows, dark_rows, log = [], [], []
    n_rows = 0
    t0 = time.time()
    with open(res / "edges_labeled.jsonl", "w", encoding="utf-8") as out:
        for k, (task, pre) in enumerate(chosen, 1):
            iid, repo = task["instance_id"], task["repo"]
            gp = find_graph(slim, task)
            if gp is None:
                log.append(f"{iid}: graph not found")
                continue
            trace = load_trace(pre)
            graph = cache.get(gp)
            mapping = map_trace(trace, graph)
            rows, st = label_edges(trace, graph, iid, repo, mapping=mapping)
            for r in rows:
                out.write(json.dumps(r) + "\n")
            n_rows += len(rows)
            cov = coverage_row(trace, graph, mapping, iid, repo)
            cov.update({f"edges_{k2}": v for k2, v in st.items()})
            cov_rows.append(cov)
            dark_rows.append(task_dark_summary(rows, iid, repo))
            if k % 10 == 0:
                print(f"  {k}/{len(chosen)} tasks, {n_rows} edge rows, {time.time() - t0:.0f}s", flush=True)

    write_csv(res / "mapping_coverage.csv", cov_rows)
    write_csv(res / "dark_by_task.csv", dark_rows)
    (res / "compare_log.txt").write_text("\n".join(log) + ("\n" if log else ""))

    # ---- console summary (paste this back)
    print(f"\nwrote {n_rows} labeled edge rows for {len(cov_rows)} tasks; skipped {len(log)}")
    by_repo: dict = {}
    for c, d in zip(cov_rows, dark_rows):
        s = by_repo.setdefault(c["repo"], {"n": 0, "cov": [], "dark": [], "pairs": 0, "darkpairs": 0, "near": 0})
        s["n"] += 1
        if c["mapped_call_fraction"] is not None:
            s["cov"].append(c["mapped_call_fraction"])
        if d["dark_fraction"] is not None:
            s["dark"].append(d["dark_fraction"])
        s["pairs"] += d["n_pairs"]
        s["darkpairs"] += d["n_dark"]
    print(f"{'repo':22s} {'tasks':>5s} {'mapped_call_frac':>17s} {'dark_frac(macro)':>17s} {'pairs':>8s} {'dark':>8s}")
    for repo, s in sorted(by_repo.items()):
        mc = sum(s["cov"]) / len(s["cov"]) if s["cov"] else float("nan")
        md = sum(s["dark"]) / len(s["dark"]) if s["dark"] else float("nan")
        print(f"{repo:22s} {s['n']:5d} {mc:17.3f} {md:17.3f} {s['pairs']:8d} {s['darkpairs']:8d}")

    # cause and near-miss breakdown of dark edges
    from collections import Counter
    causes, plens, kinds = Counter(), Counter(), Counter()
    with open(res / "edges_labeled.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["phase"] != "runtime" or r["test_code"] or r["kind"] not in ("direct", "lifted"):
                continue
            kinds[(r["kind"], "static" if r["static_edge"] else "dark")] += 1
            if not r["static_edge"]:
                causes[r["cause"]] += 1
                plens[r["static_path_len"]] += 1
    print("\nrow counts by (kind, label):", dict(kinds))
    print("dark-edge causes:", dict(causes))
    print("dark-edge static_path_len (None = >3 hops):", dict(plens))


if __name__ == "__main__":
    main()
