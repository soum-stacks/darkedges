#!/usr/bin/env python3
"""Extra 2 (HEL): task-clustered bootstrap intervals for the link-prediction AUC gaps (H3).

Rebuilds exactly the samples of step 06 (same seeds, same order) for the three variants, fits the same
leave-one-repo-out models, then resamples held-out TASKS (not edges) to get intervals for pooled AUC and for
paired differences. Point estimates should equal the AUCs in step 06's tables (sanity check).
Writes results/linkpred_bootstrap.csv. Usage: python scripts/09_linkpred_bootstrap.py [--boot 1000]
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

from darkedges import stats as S  # noqa: E402
from darkedges.dataio import GraphCache, find_graph, load_tasks, repo_root  # noqa: E402
from darkedges.linkpred import StaticCtx, build_task_samples, load_embeddings, loo_task_scores  # noqa: E402

VARIANTS = [("default", False, False), ("nohubs", True, False), ("nohubs_crossclass", True, True)]
CONTRASTS = [("emb_cos", "name_jaccard"), ("emb_cos", "same_module"), ("emb_cos", "ppr"),
             ("logreg", "logreg_no_emb")]
SHOW = ["emb_cos", "name_jaccard", "same_module", "ppr", "logreg", "logreg_no_emb"]


def build_per_task(root, by_task, tasks, ids, drop_hubs, excl_same, seed=0, max_pos=300, hits_pos=50):
    slim, emb_dir = root / "data" / "slim", root / "data" / "extras" / "embeddings"
    cache, per_task, n_emb = GraphCache(maxsize=2), [], 0
    for k, iid in enumerate(ids, 1):               # same enumeration and seeding as scripts/06_linkpred.py
        task = tasks[iid]
        gp = find_graph(slim, task)
        if gp is None:
            continue
        emb = load_embeddings(emb_dir, iid)
        n_emb += emb is not None
        ctx = StaticCtx(cache.get(gp), emb, drop_class_hubs=drop_hubs)
        rng = np.random.default_rng([seed, k])
        s = build_task_samples(ctx, by_task[iid], rng, max_pos=max_pos, hits_pos=hits_pos, exclude_same_class=excl_same)
        if len(s["y"]) == 0:
            continue
        s["repo"] = task["repo"].split("/")[-1].lower()
        s["instance_id"] = iid
        per_task.append(s)
    return per_task, n_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-tasks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo_root()
    res = root / "results"
    by_task = defaultdict(list)
    for line in open(res / "edges_labeled.jsonl", encoding="utf-8"):
        r = json.loads(line)
        by_task[r["instance_id"]].append(r)
    tasks = {t["instance_id"]: t for t in load_tasks(root / "data" / "slim")}
    ids = [i for i in by_task if i in tasks]
    if a.limit:
        ids = ids[: a.limit]

    rows = []
    for name, drop_hubs, excl_same in VARIANTS:
        t0 = time.time()
        per_task, n_emb = build_per_task(root, by_task, tasks, ids, drop_hubs, excl_same, a.seed)
        counts = Counter(t["repo"] for t in per_task)
        held = sorted(r for r, c in counts.items() if c >= a.min_tasks)
        print(f"\n=== variant {name}: {len(per_task)} tasks, held-out repos {held} ({time.time() - t0:.0f}s) ===", flush=True)
        for repo in held:
            ls = loo_task_scores(per_task, repo, use_emb=n_emb > 0, seed=a.seed)
            names = [k for k in SHOW if k in ls["scores"]]
            r = S.cluster_bootstrap_auc(ls["y"], {k: ls["scores"][k] for k in names},
                                        [c for c in CONTRASTS if c[0] in names and c[1] in names], a.boot, a.seed)
            print(f"\n{repo} ({r['n_tasks']} tasks{'; interval unreliable' if r['n_tasks'] < 10 else ''})")
            for k in names:
                v = r["auc"][k]
                print(f"  AUC {k:14s} {v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
                rows.append({"variant": name, "heldout_repo": repo, "kind": "auc", "name": k, "value": v["point"],
                             "lo": v["lo"], "hi": v["hi"], "p_diff_le0": None, "n_tasks": r["n_tasks"]})
            for c, v in r["contrasts"].items():
                print(f"  gap {c:28s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  P(diff<=0)={v['p_diff_le0']:.3f}")
                rows.append({"variant": name, "heldout_repo": repo, "kind": "diff", "name": c, "value": v["diff"],
                             "lo": v["lo"], "hi": v["hi"], "p_diff_le0": v["p_diff_le0"], "n_tasks": r["n_tasks"]})
    with open(res / "linkpred_bootstrap.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "heldout_repo", "kind", "name", "value", "lo", "hi",
                                           "p_diff_le0", "n_tasks"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {res / 'linkpred_bootstrap.csv'}")


if __name__ == "__main__":
    main()
