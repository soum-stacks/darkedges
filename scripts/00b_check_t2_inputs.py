#!/usr/bin/env python3
"""T2 preflight (HEL): verify environment + inputs and print the facts the T2 code relies on.

Run from the repo root:  python scripts/00b_check_t2_inputs.py
Paste the whole output back into the chat (it is short and contains no competition source text).
"""
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

print("python", sys.version.split()[0])
for mod in ["numpy", "scipy", "sklearn", "pandas", "matplotlib"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:11s} {m.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  {mod:11s} MISSING ({e.__class__.__name__})")

from darkedges.causes import text_has_decorators  # noqa: E402
from darkedges.compare import is_test_id, map_trace  # noqa: E402
from darkedges.dataio import find_graph, load_graph, load_tasks, load_trace, repo_root, select_tasks  # noqa: E402
from darkedges.rankings import fail_frame_nodes  # noqa: E402

root = repo_root()
slim, traces, extras = root / "data" / "slim", root / "data" / "traces", root / "data" / "extras"
print("\nroot:", root)

tasks = load_tasks(slim)
print(f"tasks: {len(tasks)}  repos: {dict(Counter(t['repo'] for t in tasks))}")
found = [(t, find_graph(slim, t)) for t in tasks]
print(f"graphs found: {sum(1 for _, g in found if g)}/{len(tasks)}")

n_pre = len(list(traces.glob("*.pre.json")))
n_post = len(list(traces.glob("*.post.json")))
print(f"\ntraces: pre={n_pre} post={n_post}  (_status.jsonl present: {(traces / '_status.jsonl').exists()})")
for ts in ("f2p", "strict"):
    sel = select_tasks(tasks, traces, ts)
    print(f"  task-set {ts:6s}: {len(sel):3d}  by repo {dict(Counter(t['repo'].split('/')[-1] for t, _ in sel))}")
print("  expected from Phase 1: f2p=114, strict=103")

# ---- one graph in detail
sel = select_tasks(tasks, traces, "strict")
if sel:
    t0, pre0 = sel[0]
    gp = find_graph(slim, t0)
    g = load_graph(gp)
    print(f"\nexample graph {gp.name}: nodes={len(g.ids)} edge types={dict(g.edge_types)}")
    print("  first node ids:", g.ids[:3])
    print(f"  test-like node ids (is_test_id): {sum(is_test_id(n) for n in g.ids)}/{len(g.ids)}")
    func_like = [n for n in g.ids if g.text[n].lstrip().startswith(("def ", "async def ", "@"))]
    print(f"  nodes whose text starts with '@' (decorators kept in text): "
          f"{sum(g.text[n].lstrip().startswith('@') for n in g.ids)}/{len(func_like)} function-like")

    # ---- mapping smoke test on up to 6 tasks
    print("\nmapping smoke test (source nodes only, call-weighted):")
    for t, pre in sel[:6]:
        gg = load_graph(find_graph(slim, t))
        tr = load_trace(pre)
        mp = map_trace(tr, gg)
        src = [n for n in tr["nodes"] if n["qualname"] != "<module>" and "<" not in n["qualname"]
               and not n["relpath"].startswith(("tests/", "test/"))]
        tot = sum(n.get("calls", 0) for n in src) or 1
        ok = sum(n.get("calls", 0) for n in src if mp.get(n["i"]))
        un = [f"{n['relpath']}:{n['qualname']}" for n in src if not mp.get(n["i"])][:2]
        print(f"  {t['instance_id']:18s} nodes={len(tr['nodes']):5d} mapped_calls={ok / tot:5.2f} unmapped e.g. {un}")

    # ---- fail_frames format
    tr = load_trace(pre0)
    ft = [k for k, v in tr["tests"].items() if v.get("outcome") in ("failed", "error")]
    print(f"\nexample task {t0['instance_id']}: tests={len(tr['tests'])} failed={len(ft)}")
    if ft:
        fr = tr["tests"][ft[0]].get("fail_frames")
        print("  fail_frames of first failing test:", fr[:4] if fr else fr)
        idx, miss = fail_frame_nodes(tr, ft)
        print(f"  frames matched to trace nodes: {len(idx)} (unmatched frames: {miss})")
        print("  test keys present:", sorted(tr["tests"][ft[0]].keys()))

# ---- extras
print("\nextras (from Kaggle step 03b):")
gp_ = extras / "gold_nodes.jsonl"
if gp_.exists():
    rows = [json.loads(x) for x in open(gp_) if x.strip()]
    print(f"  gold_nodes.jsonl rows={len(rows)} with_gold={sum(1 for r in rows if r['gold_node_ids'])}")
else:
    print("  gold_nodes.jsonl MISSING (needed for step 05)")
ei = extras / "embeddings" / "index.json"
print("  embeddings/index.json:", f"{len(json.load(open(ei)))} tasks" if ei.exists() else "MISSING (optional; needed for emb_cos)")
