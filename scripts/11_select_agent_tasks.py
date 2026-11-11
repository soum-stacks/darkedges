#!/usr/bin/env python3
"""T3.3 prep (HEL): pick the 30 agent-check tasks (protocol.yaml agent_check.tasks).

Rule (fixed before any agent run): candidates = tasks ranked in results/gold_in_trace.csv
(strict set, gold function in the graph, failing tests defined); httpx excluded (1 task);
30 tasks stratified by repo, proportional with at least 3 per repo (largest remainder);
random choice within repo with seed 0.

Reads : results/gold_in_trace.csv, data/slim/tasks.jsonl
Writes: configs/agent_check_tasks.txt (one instance_id per line; ids only, safe to commit)
Usage : python scripts/11_select_agent_tasks.py [--n 30] [--seed 0] [--root REPO]
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darkedges.dataio import load_tasks, repo_root, repo_short  # noqa: E402


def allocate(counts: dict, n: int, min_per: int = 3) -> dict:
    """Proportional allocation; repos whose share is below min_per get min(min_per, available) and the
    remainder is split proportionally among the others (largest remainder)."""
    fixed: dict = {}
    while True:
        free = {r: c for r, c in counts.items() if r not in fixed}
        left = n - sum(fixed.values())
        tot = sum(free.values())
        quota = {r: left * c / tot for r, c in free.items()} if tot else {}
        low = {r for r, q in quota.items() if q < min_per or counts[r] <= min_per}
        if not low:
            break
        for r in low:
            fixed[r] = min(counts[r], min_per)
    alloc = dict(fixed)
    for r, q in quota.items():
        alloc[r] = min(counts[r], int(q))
    while sum(alloc.values()) < n:
        opts = [r for r in quota if alloc[r] < counts[r]]
        if not opts:
            break
        r = max(opts, key=lambda r: (quota[r] - alloc[r], counts[r]))
        alloc[r] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo_root()
    git = root / "results" / "gold_in_trace.csv"
    if not git.exists():
        sys.exit(f"missing {git} (run 05_rankings first)")
    repo_of = {t["instance_id"]: repo_short(t["repo"]) for t in load_tasks(root / "data" / "slim")}
    with open(git, newline="", encoding="utf-8") as fh:
        ids = sorted({r["instance_id"] for r in csv.DictReader(fh)})
    by_repo = defaultdict(list)
    for i in ids:
        if repo_of.get(i) and repo_of[i] != "httpx":
            by_repo[repo_of[i]].append(i)
    alloc = allocate({r: len(v) for r, v in by_repo.items()}, a.n)
    rng = random.Random(a.seed)
    chosen = []
    for r in sorted(by_repo):
        chosen += sorted(rng.sample(sorted(by_repo[r]), alloc[r]))
    out = root / "configs" / "agent_check_tasks.txt"
    out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print("candidates per repo:", {r: len(v) for r, v in sorted(by_repo.items())})
    print("chosen per repo    :", dict(sorted(alloc.items())))
    print(f"wrote {out} ({len(chosen)} ids)")
    print("TASK_IDS = " + repr(chosen))


if __name__ == "__main__":
    main()
