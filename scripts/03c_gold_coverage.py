#!/usr/bin/env python
"""T1.1 acceptance + sensitivity summary. Run from the repo root (laptop or Helix) after unzipping phase1_outputs.zip:
    python scripts/03c_gold_coverage.py
Needs data/traces/_valid.jsonl (run scripts/03b_validity_report.py first), results/gold_nodes.jsonl, data/traces/_env.jsonl."""
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


valid = {r["instance_id"]: r for r in rows("data/traces/_valid.jsonl")}
gold = {}
for r in rows("results/gold_nodes.jsonl"):
    gold[r["instance_id"]] = r                       # last row wins
env = {}
for r in rows("data/traces/_env.jsonl"):
    env[r["instance_id"]] = r                        # last row wins
for key in ("valid", "valid_f2p"):
    ids = [i for i, r in valid.items() if r[key]]
    by = defaultdict(lambda: [0, 0, 0])
    for i in ids:
        repo = valid[i]["repo"]; g = gold.get(i, {})
        by[repo][0] += 1
        by[repo][1] += bool(g.get("gold_node_ids"))
        by[repo][2] += bool(env.get(i, {}).get("shims_on_path"))
    tot = [sum(v[k] for v in by.values()) for k in range(3)]
    print(f"[{key}] {tot[0]} tasks | with >=1 gold node: {tot[1]} | run with stand-in packages available: {tot[2]}")
    for repo, v in sorted(by.items()):
        print(f"    {repo:22s} tasks {v[0]:3d}  gold>=1 {v[1]:3d}  stand-ins-available {v[2]:3d}")
miss = Counter()
for i, r in valid.items():
    if r["valid"]:
        for u in gold.get(i, {}).get("unmatched", []):
            miss[u] += 1
print("most frequent unmatched gold functions (valid tasks):")
for u, n in miss.most_common(15):
    print(f"   {n:2d}  {u}")
no_gold = [i for i, r in valid.items() if r["valid"] and not gold.get(i, {}).get("gold_node_ids")]
print(f"valid tasks with NO gold node ({len(no_gold)}):", ", ".join(sorted(no_gold)[:40]))
