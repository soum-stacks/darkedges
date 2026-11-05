"""T2.1: label runtime edges against the released static graph.

Definitions (handoff Section 8):
  dark edge   = runtime edge (kind direct|lifted) between two mapped repo
                functions with no `calls` edge caller->callee in the static graph
  near-miss   = static path of length 2 (static_path_len == 2)
  phase       = 'import' if either endpoint is <module> or class-body code,
                else 'runtime'  (import-phase edges are excluded from headline
                dark-edge rates)

Mapping trace node (relpath, qualname) -> graph node id is done here, with its
own suffix-matching (src/ layouts, __init__.py), and does not import
tracelens.mapping so that this module stays testable on its own.
"""
from __future__ import annotations

from collections import defaultdict

from .causes import classify, is_class_text
from .dataio import Graph, bfs_levels

NESTED_MARKERS = ("<locals>", "<lambda>", "<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>")


# ----------------------------------------------------------------- paths
def is_test_path(relpath: str) -> bool:
    p = relpath.replace("\\", "/")
    parts = p.split("/")
    base = parts[-1]
    return (
        "tests" in parts[:-1]
        or "test" in parts[:-1]
        or base.startswith("test_")
        or base.endswith("_test.py")
        or base == "conftest.py"
    )


def is_test_id(node_id: str) -> bool:
    """Heuristic for graph node ids (dotted): any component tests/test/conftest/test_*."""
    for c in node_id.split("."):
        if c in ("tests", "test", "conftest") or c.startswith("test_"):
            return True
    return False


def module_candidates(relpath: str) -> list[str]:
    p = relpath.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    parts = [x for x in p.split("/") if x and x != "."]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts))]  # longest first


def map_node(relpath: str, qualname: str, node_set) -> str | None:
    """Graph node id for a function/method/class/module, or None."""
    if any(m in qualname for m in NESTED_MARKERS):
        return None
    for mod in module_candidates(relpath):
        cand = mod if qualname == "<module>" else f"{mod}.{qualname}"
        if cand in node_set:
            return cand
    return None


def map_trace(trace: dict, graph: Graph) -> dict[int, str | None]:
    return {n["i"]: map_node(n["relpath"], n["qualname"], graph.node_set) for n in trace["nodes"]}


def node_class(n: dict) -> str:
    """Eligibility class of a trace node: module | nested | test | source."""
    q = n["qualname"]
    if q == "<module>":
        return "module"
    if any(m in q for m in NESTED_MARKERS):
        return "nested"
    if is_test_path(n["relpath"]):
        return "test"
    return "source"


# ----------------------------------------------------------------- coverage
def coverage_row(trace: dict, graph: Graph, mapping: dict, instance_id: str, repo: str) -> dict:
    calls_all = calls_mapped = 0
    n_funcs = n_unmapped = 0
    cls_count = {"module": 0, "nested": 0, "test": 0}
    for n in trace["nodes"]:
        c = node_class(n)
        if c != "source":
            cls_count[c] += 1
            continue
        n_funcs += 1
        calls_all += n.get("calls", 0)
        if mapping.get(n["i"]) is not None:
            calls_mapped += n.get("calls", 0)
        else:
            n_unmapped += 1
    return {
        "instance_id": instance_id,
        "repo": repo,
        "mapped_call_fraction": (calls_mapped / calls_all) if calls_all else None,
        "n_funcs": n_funcs,
        "n_unmapped": n_unmapped,
        "mapped_node_fraction": ((n_funcs - n_unmapped) / n_funcs) if n_funcs else None,
        "n_module_nodes": cls_count["module"],
        "n_nested_nodes": cls_count["nested"],
        "n_test_nodes": cls_count["test"],
    }


# ----------------------------------------------------------------- labeling
def _edge_list(trace: dict, tests=None):
    """Yield (caller_i, callee_i, kind, count)."""
    if tests is None:
        for e in trace.get("edges", []):
            yield e.get("caller"), e.get("callee"), e.get("kind"), e.get("count", 1)
    else:
        seen: dict = defaultdict(int)
        for t in tests:
            for e in trace.get("tests", {}).get(t, {}).get("edges", []):
                caller, callee, kind = e[0], e[1], e[2]
                seen[(caller, callee, kind)] += 1
        for (caller, callee, kind), c in seen.items():
            yield caller, callee, kind, c


def label_edges(trace: dict, graph: Graph, instance_id: str = "", repo: str = "",
                mapping: dict | None = None, tests=None):
    """Label runtime edges. Returns (rows, stats).

    Rows are only produced for edges whose two endpoints both map to graph
    nodes. `tests` restricts to the union of the per-test edges of those tests
    (used for failing-test analyses); None uses the whole-run edge list.
    """
    mapping = mapping if mapping is not None else map_trace(trace, graph)
    by_i = {n["i"]: n for n in trace["nodes"]}
    stats = {"n_edges": 0, "n_entry": 0, "n_self": 0, "n_unmapped": 0, "n_module": 0, "n_labeled": 0}
    merged: dict = {}
    for caller, callee, kind, count in _edge_list(trace, tests):
        stats["n_edges"] += 1
        if caller is None or kind == "entry":
            stats["n_entry"] += 1
            continue
        cn, dn = by_i.get(caller), by_i.get(callee)
        if cn is None or dn is None:
            stats["n_unmapped"] += 1
            continue
        cid, did = mapping.get(caller), mapping.get(callee)
        if cid is None or did is None:
            if cn["qualname"] == "<module>" or dn["qualname"] == "<module>":
                stats["n_module"] += 1
            else:
                stats["n_unmapped"] += 1
            continue
        if cid == did:
            stats["n_self"] += 1
            continue
        key = (cid, did, kind)
        m = merged.setdefault(key, {"count": 0, "caller_i": caller, "callee_i": callee})
        m["count"] += count

    rows = []
    bfs_cache: dict = {}
    for (cid, did, kind), m in merged.items():
        cn, dn = by_i[m["caller_i"]], by_i[m["callee_i"]]
        static = graph.has_call(cid, did)
        if cid not in bfs_cache:
            bfs_cache[cid] = bfs_levels(graph.calls_out, [cid], 3)
        plen = bfs_cache[cid].get(did)
        phase = "import" if (
            cn["qualname"] == "<module>" or dn["qualname"] == "<module>"
            or is_class_text(graph.text.get(cid, "")) or is_class_text(graph.text.get(did, ""))
        ) else "runtime"
        cause = None if static else classify(graph.text.get(did, ""), kind, name=graph.name.get(did))
        rows.append({
            "instance_id": instance_id, "repo": repo,
            "caller_id": cid, "callee_id": did, "kind": kind, "count": m["count"],
            "phase": phase, "static_edge": static, "static_path_len": plen,
            "cause": cause, "any_type_edge": (cid, did) in graph.any_pairs,
            "test_code": is_test_path(cn["relpath"]) or is_test_path(dn["relpath"]),
        })
    stats["n_labeled"] = len(rows)
    return rows, stats


# ----------------------------------------------------------------- aggregates
def task_dark_summary(rows: list[dict], instance_id: str = "", repo: str = "") -> dict:
    """Per-task dark-edge fractions over unique (caller, callee) pairs.

    Headline set: phase == 'runtime', test_code == False, kind in direct|lifted.
    'direct' sensitivity: pairs that have at least one direct row.
    """
    pairs: dict = {}
    imp: dict = {}
    for r in rows:
        if r["kind"] not in ("direct", "lifted") or r["test_code"]:
            continue
        tgt = pairs if r["phase"] == "runtime" else imp
        p = tgt.setdefault((r["caller_id"], r["callee_id"]),
                           {"dark": not r["static_edge"], "direct": False, "count": 0})
        p["direct"] = p["direct"] or r["kind"] == "direct"
        p["count"] += r["count"]

    def frac(d, sel=None):
        items = [v for v in d.values() if sel is None or sel(v)]
        n = len(items)
        k = sum(1 for v in items if v["dark"])
        return n, k, (k / n if n else None)

    n, k, f = frac(pairs)
    nd, kd, fd = frac(pairs, lambda v: v["direct"])
    ni, ki, fi = frac(imp)
    cnt_all = sum(v["count"] for v in pairs.values())
    cnt_dark = sum(v["count"] for v in pairs.values() if v["dark"])
    return {
        "instance_id": instance_id, "repo": repo,
        "n_pairs": n, "n_dark": k, "dark_fraction": f,
        "n_pairs_direct": nd, "n_dark_direct": kd, "dark_fraction_direct": fd,
        "n_pairs_import": ni, "n_dark_import": ki, "dark_fraction_import": fi,
        "dark_fraction_count_weighted": (cnt_dark / cnt_all) if cnt_all else None,
    }
