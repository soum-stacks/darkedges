#!/usr/bin/env python3
"""T2.2 (HEL): localisation rankings (ochiai, stack_dist, static_k1-3, bm25) and gold-in-trace.

Reads : data/slim/*, data/traces/<id>.pre.json, data/extras/gold_nodes.jsonl
Writes: results/rankings.csv, results/gold_in_trace.csv, results/task_summary.csv
Run   : python scripts/05_rankings.py [--task-set strict|f2p|all] [--shuffles 100] [--limit N]
Tasks without any gold node that exists in the graph are skipped (listed in results/rankings_skipped.txt).
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkedges.dataio import GraphCache, find_graph, load_tasks, load_trace, read_jsonl, repo_root, select_tasks  # noqa: E402
from darkedges.rankings import BM25, KS, build_name_index, candidate_ok, rank_task, tokenize  # noqa: E402


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-set", default="strict", choices=["strict", "f2p", "all"])
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    root = Path(a.root).resolve() if a.root else repo_root()
    slim, traces, res = root / "data" / "slim", root / "data" / "traces", root / "results"
    gold_path = root / "data" / "extras" / "gold_nodes.jsonl"
    res.mkdir(exist_ok=True)
    if not gold_path.exists():
        sys.exit(f"missing {gold_path}: run scripts/03b_gold_and_embeddings.py on Kaggle first (see runbook)")
    gold = {r["instance_id"]: set(r["gold_node_ids"]) for r in read_jsonl(gold_path)}

    tasks = load_tasks(slim)
    chosen = select_tasks(tasks, traces, a.task_set)
    if a.limit:
        chosen = chosen[: a.limit]
    print(f"task set '{a.task_set}': {len(chosen)} tasks")

    cache = GraphCache(maxsize=4)
    all_rows, summaries, gits, skipped = [], [], [], []
    bm_cache: dict = {}
    t0 = time.time()
    for k, (task, pre) in enumerate(chosen, 1):
        iid = task["instance_id"]
        gp = find_graph(slim, task)
        if gp is None:
            skipped.append(f"{iid}: graph not found")
            continue
        graph = cache.get(gp)
        g = {x for x in gold.get(iid, set()) if x in graph.node_set}
        if not g:
            skipped.append(f"{iid}: no gold node in graph (gold_nodes={len(gold.get(iid, set()))})")
            continue
        if gp not in bm_cache:
            bm_cache.clear()
            docs = {n: tokenize(graph.text[n]) for n in graph.ids if candidate_ok(n)}
            bm_cache[gp] = (BM25(docs), build_name_index(graph))
        bm, nidx = bm_cache[gp]
        rows, summ, git = rank_task(task, load_trace(pre), graph, g, bm25=bm, name_index=nidx,
                                    n_shuffles=a.shuffles)
        all_rows += rows
        summaries.append(summ)
        gits.append(git)
        if k % 10 == 0:
            print(f"  {k}/{len(chosen)} tasks, {time.time() - t0:.0f}s", flush=True)

    write_csv(res / "rankings.csv", all_rows)
    write_csv(res / "gold_in_trace.csv", gits)
    write_csv(res / "task_summary.csv", summaries)
    (res / "rankings_skipped.txt").write_text("\n".join(skipped) + ("\n" if skipped else ""))

    # ---- console summary (paste this back)
    print(f"\nranked {len(gits)} tasks; skipped {len(skipped)}")
    agg = defaultdict(list)
    for r in all_rows:
        agg[(r["method"], r["K"])].append(r["hit"])
    print("recall@K (mean over tasks)")
    print(f"{'method':12s}" + "".join(f"{'K=' + str(k):>8s}" for k in KS))
    for m in ["ochiai", "stack_dist", "static_k1", "static_k2", "static_k3", "bm25"]:
        print(f"{m:12s}" + "".join(f"{sum(agg[(m, k)]) / max(1, len(agg[(m, k)])):8.3f}" for k in KS))
    n = max(1, len(gits))
    print(f"\ngold executed by failing tests (in_trace):   {sum(g['in_trace'] for g in gits)}/{n}")
    print(f"  ...and not by passing tests (in_trace_diff): {sum(g['in_trace_diff'] for g in gits)}/{n}")
    print(f"  ...only via dark edges (dark_only):          {sum(g['dark_only'] for g in gits)}/{n}")
    print(f"gold within static 3 hops of a failing frame: {sum(g['gold_in_static_k3_of_fail'] for g in gits)}/{n}")
    print(f"tasks with no issue-text seeds: {sum(1 for s in summaries if s['n_seeds'] == 0)}/{n}")
    print(f"tasks where Ochiai has no passing tests: {sum(1 for s in summaries if s['n_passed'] == 0)}/{n}")


if __name__ == "__main__":
    main()
