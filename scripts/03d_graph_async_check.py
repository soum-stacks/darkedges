#!/usr/bin/env python
"""Do the released static graphs contain async functions? + why do some valid tasks have no gold node?
Run from the repo root (laptop or Helix):  python scripts/03d_graph_async_check.py [--graphs data/slim]
Private output (competition-derived): do not paste node text anywhere public."""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--graphs", default="data/slim")
a = ap.parse_args()
DEF = re.compile(r"^\s*(async\s+def|def|class)\s+\w+")


def kind(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("@", "#")):
            continue
        m = DEF.match(line)
        return m.group(1).split()[0] if m and m.group(1).startswith("async") else (m.group(1) if m else "other")
    return "other"


seen, per = set(), defaultdict(Counter)
for f in sorted(Path(a.graphs).rglob("*.json")):
    if f.stat().st_size < 100 or f.name in seen:
        continue
    seen.add(f.name)
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    repo = f.name.split("_")[0]
    for n in d.get("nodes", []):
        per[repo][kind(n.get("text", ""))] += 1
print("graph node kinds by first definition keyword in node text (distinct graph files):")
for repo, c in sorted(per.items()):
    print(f"  {repo:10s} {dict(c)}")
print("=> async nodes:", sum(c.get('async', 0) for c in per.values()))

def rows(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []
valid = {r["instance_id"]: r for r in rows("data/traces/_valid.jsonl")}
gold = {r["instance_id"]: r for r in rows("results/gold_nodes.jsonl")}
print("\nvalid tasks with no gold node:  id | n_hunks | unmapped_hunks | n_unmatched_functions | note")
for i in sorted(valid):
    if valid[i]["valid"] and not gold.get(i, {}).get("gold_node_ids"):
        g = gold.get(i, {})
        note = "function not in graph (e.g. async)" if g.get("unmatched") else "module/class-level or non-function edit"
        print(f"  {i:16s} {g.get('n_hunks')} | {g.get('unmapped_hunks')} | {len(g.get('unmatched', []))} | {note}")