#!/usr/bin/env python3
"""T2.0 (KCPU): gold nodes + embeddings, extracted from the competition data.

SELF-CONTAINED (stdlib + numpy). Paste the whole file into ONE cell of a Kaggle
notebook (competition attached, no accelerator, internet off) and run it.
Outputs /kaggle/working/t2_extras.zip containing:
    gold_nodes.jsonl          one row per task (spec in handoff Section 7.4)
    embeddings/<stem>.npz     arrays ids (str) and emb (float16, n x 256), one per unique graph
    embeddings/index.json     {instance_id: stem}
    extract_report.txt        everything the console printed
Competition-derived: keep PRIVATE (Rule 4b).

Gold nodes (handoff Section 9): functions/methods in the BASE file whose line span
contains a changed OLD-side line of the gold patch. Pure additions are anchored to
the old line just before the insertion and counted for function F only if the
first added line is indented deeper than F's `def`. If the innermost function has
no graph node (e.g. nested function), the nearest enclosing function/method that
has one is used. Hunks that map to no graph node count as unmapped_hunks.
"""
import ast
import io
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ settings
COMP_CANDIDATES = [
    "/kaggle/input/competitions/gemma-4-developer-agent",
    "/kaggle/input/gemma-4-developer-agent",
]
OUT_DIR = "/kaggle/working/t2_extras"
DO_EMBEDDINGS = True      # set False to skip embeddings (gold nodes only)
EMB_DTYPE = np.float16


# ------------------------------------------------------------------ mapping (same logic as darkedges.compare.map_node)
def module_candidates(relpath):
    p = relpath.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    parts = [x for x in p.split("/") if x and x != "."]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts))]


def map_id(relpath, qualname, node_set):
    for mod in module_candidates(relpath):
        cand = mod if qualname == "<module>" else f"{mod}.{qualname}"
        if cand in node_set:
            return cand
    return None


# ------------------------------------------------------------------ unified diff
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _strip_path(s):
    s = s.split("\t")[0].strip()
    if s == "/dev/null":
        return None
    if s.startswith(("a/", "b/")):
        s = s[2:]
    return s


def parse_diff(patch_text):
    """-> list of {old_path, new_path, hunks:[items]}; items are
    ('del', old_line) or ('add', anchor_old_line, indent)."""
    files, cur, hunk = [], None, None
    lines = patch_text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("diff --git"):
            cur = {"old_path": None, "new_path": None, "hunks": []}
            files.append(cur)
            hunk = None
        elif cur is not None and hunk is None and ln.startswith("--- "):
            cur["old_path"] = _strip_path(ln[4:])
        elif cur is not None and hunk is None and ln.startswith("+++ "):
            cur["new_path"] = _strip_path(ln[4:])
        else:
            m = _HUNK.match(ln)
            if m and cur is not None:
                old = int(m.group(1))
                rem_old = int(m.group(2)) if m.group(2) is not None else 1
                rem_new = int(m.group(4)) if m.group(4) is not None else 1
                items, in_add = [], False
                i += 1
                while i < len(lines) and (rem_old > 0 or rem_new > 0):
                    h = lines[i]
                    tag = h[:1] if h else " "
                    if tag == "\\":
                        i += 1
                        continue
                    if tag == "-":
                        items.append(("del", old))
                        old += 1
                        rem_old -= 1
                        in_add = False
                    elif tag == "+":
                        body = h[1:]
                        if not in_add:
                            items.append(["add", old - 1, None])
                            in_add = True
                        if items[-1][2] is None and body.strip():
                            items[-1][2] = len(body) - len(body.lstrip(" \t"))
                        rem_new -= 1
                    else:
                        old += 1
                        rem_old -= 1
                        rem_new -= 1
                        in_add = False
                    i += 1
                cur["hunks"].append([tuple(x) for x in items])
                hunk = None
                continue
        i += 1
    return files


# ------------------------------------------------------------------ function spans
def collect_spans(tree):
    out = []

    def visit(node, chain):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                c2 = chain + [ch.name]
                out.append({"start": start, "end": ch.end_lineno, "col": ch.col_offset,
                            "chain": c2, "is_func": not isinstance(ch, ast.ClassDef)})
                visit(ch, c2)
            else:
                visit(ch, chain)

    visit(tree, [])
    return out


def _resolve(spans, line, relpath, node_set, indent=None):
    cands = [s for s in spans if s["is_func"] and s["start"] <= line <= s["end"]
             and (indent is None or indent > s["col"])]
    cands.sort(key=lambda s: s["end"] - s["start"])
    for s in cands:  # innermost first, then enclosing functions
        chain = s["chain"]
        for j in range(len(chain), 0, -1):
            nid = map_id(relpath, ".".join(chain[:j]), node_set)
            if nid:
                return nid
    return None


def gold_for_patch(patch_text, read_file, node_set):
    """read_file(path) -> source text of the BASE file or None."""
    gold, n_hunks, unmapped, n_py, n_nonpy, notes = set(), 0, 0, 0, 0, []
    for f in parse_diff(patch_text):
        path = f["old_path"]
        if path is None:           # new file: nothing existed at base
            continue
        if not path.endswith(".py"):
            n_nonpy += 1
            continue
        n_py += 1
        src = read_file(path)
        if src is None:
            notes.append(f"base file missing: {path}")
            n_hunks += len(f["hunks"])
            unmapped += len(f["hunks"])
            continue
        try:
            spans = collect_spans(ast.parse(src))
        except SyntaxError as e:
            notes.append(f"syntax error in {path}: {e}")
            n_hunks += len(f["hunks"])
            unmapped += len(f["hunks"])
            continue
        for items in f["hunks"]:
            n_hunks += 1
            found = set()
            for it in items:
                if it[0] == "del":
                    nid = _resolve(spans, it[1], path, node_set)
                else:
                    nid = _resolve(spans, it[1], path, node_set, indent=it[2]) if it[2] is not None else None
                if nid:
                    found.add(nid)
            if found:
                gold |= found
            else:
                unmapped += 1
    return {"gold_node_ids": sorted(gold), "n_hunks": n_hunks, "unmapped_hunks": unmapped,
            "n_py_files": n_py, "n_nonpy_files": n_nonpy, "notes": notes}


# ------------------------------------------------------------------ competition I/O
def find_comp():
    for c in COMP_CANDIDATES:
        if os.path.isdir(c):
            return Path(c)
    root = Path("/kaggle/input")
    for p in root.rglob("tasks.jsonl"):
        return p.parent
    raise SystemExit("competition dir not found; contents of /kaggle/input: "
                     + str(os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "n/a"))


def find_file(comp, names, subdirs=("", "graphs", "snapshots", "embeddings")):
    for sd in subdirs:
        for n in names:
            p = comp / sd / n if sd else comp / n
            if p.exists():
                return p
    return None


def short(repo):
    return repo.split("/")[-1].lower()


def find_graph(comp, task):
    iid, base = task["instance_id"], task["base_commit"]
    p = find_file(comp, [f"{short(task['repo'])}_{base}.json", f"{iid}.json"], ("graphs", ""))
    if p is not None and p.stat().st_size > 100:
        return p
    for q in (comp / "graphs").glob(f"*{base}*.json"):
        if q.stat().st_size > 100:
            return q
    return None


def make_tar_reader(tgz_path):
    tar = tarfile.open(tgz_path, "r:*")
    names = {}
    for m in tar.getmembers():
        if m.isfile():
            n = m.name[2:] if m.name.startswith("./") else m.name
            names[n] = m

    def read(path):
        m = names.get(path)
        if m is None:
            hits = [n for n in names if n.endswith("/" + path)]
            if not hits:
                return None
            m = names[min(hits, key=len)]
        data = tar.extractfile(m).read()
        return data.decode("utf-8", errors="replace")

    return read, tar


def load_node_ids(graph_path):
    d = json.load(open(graph_path, encoding="utf-8"))
    return {n["id"] for n in d["nodes"]}


def find_embedding(comp, task):
    edir = comp / "embeddings"
    if not edir.is_dir():
        return None
    iid, base = task["instance_id"], task["base_commit"]
    for n in (f"{short(task['repo'])}_{base}.npz", f"{iid}.npz"):
        if (edir / n).exists():
            return edir / n
    hits = sorted(edir.glob(f"*{base[:7]}*.npz")) or sorted(edir.glob(f"*{base[:7]}*"))
    return hits[0] if hits else None


def convert_embedding(src_path, node_set, dst_path):
    z = np.load(src_path, allow_pickle=False)
    ids, rows = [], []
    for k in z.files:
        nid = k[:-4] if k.endswith(".npy") else k
        if nid in node_set:
            ids.append(nid)
            rows.append(np.asarray(z[k]).reshape(-1))
    if not ids:
        return 0
    np.savez_compressed(dst_path, ids=np.array(ids), emb=np.stack(rows).astype(EMB_DTYPE))
    return len(ids)


# ------------------------------------------------------------------ main
def main():
    log = io.StringIO()

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    comp = find_comp()
    P("competition dir:", comp)
    P("top level:", sorted(os.listdir(comp))[:20])
    tasks_path = find_file(comp, ["tasks.jsonl"], ("", "data"))
    if tasks_path is None:
        hits = list(comp.rglob("tasks.jsonl"))
        tasks_path = hits[0] if hits else None
    if tasks_path is None:
        raise SystemExit("tasks.jsonl not found under " + str(comp))
    tasks = [json.loads(x) for x in open(tasks_path, encoding="utf-8") if x.strip()]
    P("tasks:", len(tasks), "from", tasks_path)
    snap_dir = comp / "snapshots"
    emb_dir = comp / "embeddings"
    P("snapshots dir exists:", snap_dir.is_dir(), "| embeddings dir exists:", emb_dir.is_dir())
    if emb_dir.is_dir():
        P("first embedding files:", sorted(os.listdir(emb_dir))[:5])

    out = Path(OUT_DIR)
    if out.exists():
        shutil.rmtree(out)
    (out / "embeddings").mkdir(parents=True)

    gold_rows, emb_index, emb_done = [], {}, {}
    status = {"ok": 0, "no_snapshot": 0, "no_graph": 0}
    total_emb_bytes = 0
    for k, t in enumerate(tasks, 1):
        iid = t["instance_id"]
        gp = find_graph(comp, t)
        row = {"instance_id": iid, "gold_node_ids": [], "n_hunks": 0, "unmapped_hunks": 0, "status": "ok", "notes": []}
        if gp is None:
            row["status"] = "no_graph"
            status["no_graph"] += 1
            gold_rows.append(row)
            continue
        node_set = load_node_ids(gp)
        snap = snap_dir / f"{iid}.tgz"
        if not snap.exists():
            row["status"] = "no_snapshot"
            status["no_snapshot"] += 1
        else:
            read, tar = make_tar_reader(snap)
            try:
                r = gold_for_patch(t["patch"], read, node_set)
            finally:
                tar.close()
            row.update(r)
            status["ok"] += 1

        if DO_EMBEDDINGS:
            stem = f"{short(t['repo'])}_{t['base_commit']}"
            if stem not in emb_done:
                ep = find_embedding(comp, t)
                n = convert_embedding(ep, node_set, out / "embeddings" / f"{stem}.npz") if ep else 0
                emb_done[stem] = n
                if n:
                    total_emb_bytes += (out / "embeddings" / f"{stem}.npz").stat().st_size
            if emb_done[stem]:
                emb_index[iid] = stem
        gold_rows.append(row)
        if k % 20 == 0:
            P(f"  {k}/{len(tasks)} tasks")

    with open(out / "gold_nodes.jsonl", "w", encoding="utf-8") as fh:
        for r in gold_rows:
            fh.write(json.dumps(r) + "\n")
    json.dump(emb_index, open(out / "embeddings" / "index.json", "w"))

    # ---- report (paste this back)
    n_with = sum(1 for r in gold_rows if r["gold_node_ids"])
    P("\n=== GOLD NODES ===")
    P("status:", status)
    P(f"tasks with >=1 gold node: {n_with}/{len(gold_rows)}")
    by_repo = {}
    for t, r in zip(tasks, gold_rows):
        s = by_repo.setdefault(short(t["repo"]), [0, 0, 0, 0])
        s[0] += 1
        s[1] += bool(r["gold_node_ids"])
        s[2] += r["n_hunks"]
        s[3] += r["unmapped_hunks"]
    for repo, (n, w, h, u) in sorted(by_repo.items()):
        P(f"  {repo:10s} tasks={n:3d} with_gold={w:3d} hunks={h:4d} unmapped_hunks={u:4d}")
    ng = [len(r["gold_node_ids"]) for r in gold_rows if r["gold_node_ids"]]
    if ng:
        P(f"gold nodes per task (with gold): min={min(ng)} median={sorted(ng)[len(ng) // 2]} max={max(ng)}")
    bad = [r for r in gold_rows if not r["gold_node_ids"]]
    P("first tasks without gold:")
    for r in bad[:8]:
        P("  ", r["instance_id"], r["status"], r.get("n_hunks"), r.get("unmapped_hunks"), r.get("notes", [])[:1])
    P("example:", json.dumps(next((r for r in gold_rows if r["gold_node_ids"]), {}))[:300])

    P("\n=== EMBEDDINGS ===")
    if DO_EMBEDDINGS:
        P(f"unique graphs with embeddings: {sum(1 for v in emb_done.values() if v)}/{len(emb_done)}")
        P(f"tasks indexed: {len(emb_index)}/{len(tasks)}; total size {total_emb_bytes / 1e6:.1f} MB")
        if emb_done:
            ex = next(iter(emb_done))
            P("example nodes in first embedding file:", emb_done[ex])
    else:
        P("skipped (DO_EMBEDDINGS=False)")

    (out / "extract_report.txt").write_text(log.getvalue())
    zpath = shutil.make_archive("/kaggle/working/t2_extras", "zip", OUT_DIR)
    P(f"\nwrote {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
