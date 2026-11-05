#!/usr/bin/env python3
"""T2.3 (HEL): predict dark edges; leave-one-repo-out.

Reads : results/edges_labeled.jsonl (from 04), data/slim/*, data/extras/embeddings/ (optional)
Writes: results/linkpred.csv, results/linkpred_notes.txt
Run   : python scripts/06_linkpred.py [--max-pos 300] [--seed 0] [--limit N]
Repos with fewer than --min-tasks tasks (httpx) are used for training only, never held out.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkedges.dataio import GraphCache, find_graph, load_tasks, repo_root  # noqa: E402
from darkedges.linkpred import SCORERS, StaticCtx, build_task_samples, evaluate_loo, load_embeddings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pos", type=int, default=300)
    ap.add_argument("--hits-pos", type=int, default=50)
    ap.add_argument("--min-tasks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    root = Path(a.root).resolve() if a.root else repo_root()
    slim, res = root / "data" / "slim", root / "results"
    emb_dir = root / "data" / "extras" / "embeddings"
    edges_path = res / "edges_labeled.jsonl"
    if not edges_path.exists():
        sys.exit("results/edges_labeled.jsonl missing: run scripts/04_compare_static.py first")

    by_task = defaultdict(list)
    with open(edges_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_task[r["instance_id"]].append(r)
    tasks = {t["instance_id"]: t for t in load_tasks(slim)}
    ids = [i for i in by_task if i in tasks]
    if a.limit:
        ids = ids[: a.limit]
    print(f"{len(ids)} tasks with labeled edges")

    cache = GraphCache(maxsize=2)
    per_task, notes = [], []
    n_emb = 0
    t0 = time.time()
    for k, iid in enumerate(ids, 1):
        task = tasks[iid]
        gp = find_graph(slim, task)
        if gp is None:
            notes.append(f"{iid}: graph not found")
            continue
        graph = cache.get(gp)
        emb = load_embeddings(emb_dir, iid)
        n_emb += emb is not None
        ctx = StaticCtx(graph, emb)
        rng = np.random.default_rng([a.seed, k])
        s = build_task_samples(ctx, by_task[iid], rng, max_pos=a.max_pos, hits_pos=a.hits_pos)
        if len(s["y"]) == 0:
            notes.append(f"{iid}: no dark positives")
            continue
        s["repo"] = task["repo"].split("/")[-1].lower()
        s["instance_id"] = iid
        per_task.append(s)
        if k % 10 == 0:
            print(f"  {k}/{len(ids)} tasks, {time.time() - t0:.0f}s", flush=True)

    use_emb = n_emb > 0
    if not use_emb:
        notes.append("no embeddings found: emb_cos and logreg (with embeddings) omitted")
    elif n_emb < len(ids):
        notes.append(f"embeddings found for {n_emb}/{len(ids)} tasks; missing ones get cosine 0.0")

    counts = Counter(t["repo"] for t in per_task)
    held = sorted(r for r, c in counts.items() if c >= a.min_tasks)
    print("tasks with positives per repo:", dict(counts), "| held-out repos:", held)
    rows = evaluate_loo(per_task, held, use_emb=use_emb, seed=a.seed)

    with open(res / "linkpred.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["heldout_repo", "scorer", "auc", "ap", "hits10", "n_pos", "n_neg"])
        w.writeheader()
        w.writerows(rows)
    (res / "linkpred_notes.txt").write_text("\n".join(notes) + ("\n" if notes else ""))

    # ---- console summary (paste this back)
    print(f"\n{'held-out':10s} {'scorer':18s} {'AUC':>6s} {'AP':>6s} {'Hits@10':>8s} {'n_pos':>6s} {'n_neg':>7s}")
    for r in rows:
        h = "  n/a" if r["hits10"] is None else f"{r['hits10']:8.3f}"
        print(f"{r['heldout_repo']:10s} {r['scorer']:18s} {r['auc']:6.3f} {r['ap']:6.3f} {h:>8s} {r['n_pos']:6d} {r['n_neg']:7d}")
    if notes:
        print("\nnotes:", *notes[:10], sep="\n  ")


if __name__ == "__main__":
    main()
