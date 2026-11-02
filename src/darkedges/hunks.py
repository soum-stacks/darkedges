"""Gold nodes: functions of the BASE files whose line span overlaps a changed OLD-side line
of the gold patch (Handoff Pack Sections 8-9).

Output row for results/gold_nodes.jsonl:
  {"instance_id", "gold_node_ids": [...], "n_hunks": int, "unmapped_hunks": int, ...}
Only .py files, test files excluded by the caller (the gold `patch` has none by construction).
Pure insertions are attributed to the function that encloses the insertion point
(previous old line inside F, and next old line inside F or first added line indented deeper than F's def).
Changes to nested functions are attributed to the enclosing non-nested function (the tracer maps
<locals> functions to no graph node by design).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch_text: str) -> list:
    """-> [{"path": old-side relpath, "hunks": [{"removed": [old line nos], "inserts": [(after_old_line, first_added_text)]}]}]"""
    files, cur, hunk = [], None, None
    old_no = 0
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            path = line[4:].split("\t")[0].strip()
            cur = {"path": None if path == "/dev/null" else re.sub(r"^a/", "", path), "hunks": []}
            files.append(cur); hunk = None
            continue
        if line.startswith("+++ ") and cur is not None:
            new = line[4:].split("\t")[0].strip()
            if cur["path"] is None:                      # new file: nothing on the old side
                cur["path"] = re.sub(r"^b/", "", new); cur["is_new"] = True
            continue
        m = _HUNK.match(line)
        if m and cur is not None:
            hunk = {"removed": [], "inserts": []}
            cur["hunks"].append(hunk)
            old_no = int(m.group(1))
            if m.group(2) == "0":          # zero-length old side: start = line BEFORE the insertion
                old_no += 1
            continue
        if hunk is None or cur is None:
            continue
        if line.startswith("\\"):
            continue
        if line.startswith("-"):
            hunk["removed"].append(old_no); old_no += 1
        elif line.startswith("+"):
            # insertion point: after old line (old_no - 1), before old_no
            if not hunk.get("_run"):          # first line of a run of added lines
                hunk["inserts"].append((old_no - 1, line[1:]))
            hunk["_run"] = True
            continue
        else:
            old_no += 1
        hunk["_run"] = False
    return files


def function_spans(source: str) -> list:
    """-> [(start, end, qualname, def_col, nested)] for every function/method (decorators included)."""
    tree = ast.parse(source)
    out = []

    def visit(node, prefix, in_func):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + ch.name
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.append((start, ch.end_lineno, q, ch.col_offset, in_func))
                visit(ch, q + ".<locals>.", True)
            elif isinstance(ch, ast.ClassDef):
                visit(ch, prefix + ch.name + ".", in_func)
            else:
                visit(ch, prefix, in_func)

    visit(tree, "", False)
    return out


def _innermost(spans, line):
    """Innermost NON-nested function covering `line` (nested ones attribute to the enclosing one)."""
    best = None
    for s, e, q, col, nested in spans:
        if nested or not (s <= line <= e):
            continue
        if best is None or (e - s) < (best[1] - best[0]):
            best = (s, e, q, col)
    return best


def default_to_node_id(relpath: str, qualname: str, node_ids) -> str | None:
    """File path + qualname -> graph node id (dotted symbol path). Tries src/ layouts."""
    mod = relpath[:-3] if relpath.endswith(".py") else relpath
    parts = mod.split("/")
    cands = []
    for drop in (0, 1):
        if drop and parts and parts[0] in ("src", "lib"):
            p = parts[1:]
        elif drop:
            continue
        else:
            p = parts
        if p and p[-1] == "__init__":
            p = p[:-1]
        cands.append(".".join(p + [qualname]))
    for c in cands:
        if c in node_ids:
            return c
    return None


def gold_nodes_detailed(patch_text: str, repo_dir, node_ids, mapper=default_to_node_id) -> dict:
    repo = Path(repo_dir)
    gold, n_hunks, unmapped, unmatched = set(), 0, 0, []
    for f in parse_patch(patch_text):
        path = f["path"]
        if not path or not path.endswith(".py") or f.get("is_new"):
            continue
        src_file = repo / path
        try:
            source = src_file.read_text(encoding="utf-8", errors="replace")
            spans = function_spans(source)
        except (OSError, SyntaxError):
            n_hunks += len(f["hunks"]); unmapped += len(f["hunks"]); continue
        lines = source.splitlines()
        for h in f["hunks"]:
            n_hunks += 1
            quals = set()
            for ln in h["removed"]:
                s = _innermost(spans, ln)
                if s:
                    quals.add(s[2])
            for after, added in h["inserts"]:
                prev = _innermost(spans, after) if after >= 1 else None
                nxt = _innermost(spans, after + 1) if after + 1 <= len(lines) else None
                if prev and ((nxt and nxt[2] == prev[2]) or
                             (len(added) - len(added.lstrip()) > prev[3] and added.strip())):
                    quals.add(prev[2])
            ids = set()
            for q in quals:
                nid = mapper(path, q, node_ids)
                if nid:
                    ids.add(nid)
                else:
                    unmatched.append(f"{path}::{q}")
            if ids:
                gold |= ids
            else:
                unmapped += 1
    return {"gold_node_ids": sorted(gold), "n_hunks": n_hunks, "unmapped_hunks": unmapped,
            "unmatched": sorted(set(unmatched))}


def gold_nodes(patch_text: str, repo_dir, node_ids, mapper=default_to_node_id) -> set:
    return set(gold_nodes_detailed(patch_text, repo_dir, node_ids, mapper)["gold_node_ids"])
