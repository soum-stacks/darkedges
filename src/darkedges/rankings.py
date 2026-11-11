"""T2.2: localisation rankings and the gold-in-trace ceiling.

Methods (handoff Section 8): ochiai, stack_dist, static_k1/2/3, bm25.

Conventions (documented design choices, see docs/t2_runbook.md):
  * Candidates are graph nodes that are not test code (is_test_id).
  * A method only ranks nodes it has evidence for (score > 0 / reachable);
    nodes without evidence are never returned, so n_candidates differs by method.
  * Ties are broken randomly; every statistic is averaged over n_shuffles
    shuffles (default 100, seeded from the instance id) so `hit` is in [0, 1];
    `hit_bin` = hit >= 0.5 is provided for paired tests (McNemar).
  * stack_dist = undirected shortest path in the runtime graph of the failing
    tests from the nearest traceback frame (multi-source BFS).
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter, defaultdict

import numpy as np

from .compare import is_test_id, is_test_path, label_edges, map_trace
from .dataio import Graph, bfs_levels, failed_tests, passed_tests

KS = (5, 10, 20, 50)
METHODS = ("ochiai", "stack_dist", "static_k1", "static_k2", "static_k3", "bm25")


# ----------------------------------------------------------------- text
def split_identifier(s: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.replace("_", " ").lower().split()


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""):
        subs = split_identifier(ident)
        toks.extend(t for t in subs if len(t) > 1)
        if len(subs) > 1:
            toks.append(ident.lower())
    return toks


class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = {d: len(t) for d, t in docs.items()}
        self.avgdl = (sum(self.dl.values()) / self.N) if self.N else 0.0
        self.inv: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for d, toks in docs.items():
            for term, tf in Counter(toks).items():
                self.inv[term].append((d, tf))

    def scores(self, query_tokens: list[str]) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for term in set(query_tokens):
            postings = self.inv.get(term)
            if not postings:
                continue
            n = len(postings)
            idf = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
            for d, tf in postings:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[d] / (self.avgdl or 1.0))
                out[d] += idf * tf * (self.k1 + 1) / denom
        return {d: s for d, s in out.items() if s > 0}


# ----------------------------------------------------------------- candidates
def candidate_ok(node_id: str) -> bool:
    return not is_test_id(node_id)


def eligible_map(trace: dict, mapping: dict) -> dict[int, str]:
    """trace node index -> graph id, only for mapped, non-test candidate nodes."""
    out = {}
    for n in trace["nodes"]:
        nid = mapping.get(n["i"])
        if nid is None or not candidate_ok(nid) or is_test_path(n["relpath"]):
            continue
        out[n["i"]] = nid
    return out


def mapped_ids_of_nodes(trace: dict, mapping: dict, idxs, elig: dict | None = None) -> set[str]:
    elig = elig if elig is not None else eligible_map(trace, mapping)
    return {elig[i] for i in idxs if i in elig}


# ----------------------------------------------------------------- dynamic methods
def ochiai_scores(trace: dict, mapping: dict) -> tuple[dict[str, float], dict]:
    tests = trace.get("tests", {})
    ftests, ptests = failed_tests(trace), passed_tests(trace)
    elig = eligible_map(trace, mapping)
    f, p = Counter(), Counter()
    for t in ftests:
        for nid in mapped_ids_of_nodes(trace, mapping, tests[t].get("nodes", []), elig):
            f[nid] += 1
    for t in ptests:
        for nid in mapped_ids_of_nodes(trace, mapping, tests[t].get("nodes", []), elig):
            p[nid] += 1
    total_failed = len(ftests)
    scores = {}
    for nid, fi in f.items():
        scores[nid] = fi / math.sqrt(total_failed * (fi + p.get(nid, 0)))
    return scores, {"n_failed": total_failed, "n_passed": len(ptests)}


def fail_frame_nodes(trace: dict, ftests: list[str]) -> tuple[set[int], int]:
    """Trace node indices matching the traceback frames of the failing tests.

    fail_frames entries are [relpath, lineno, func]; matched to the node in the
    same file whose last qualname component == func and firstlineno <= lineno
    (nearest). Returns (indices, n_frames_unmatched).
    """
    by_file: dict[str, list] = defaultdict(list)
    for n in trace["nodes"]:
        by_file[n["relpath"]].append(n)
    frames = []
    for t in ftests:
        frames.extend(trace["tests"][t].get("fail_frames", []) or [])
    if not frames:
        frames = trace.get("script_fail_frames", []) or []
    found, missing = set(), 0
    for fr in frames:
        relpath, lineno, func = fr[0], fr[1], fr[2]
        best = None
        for n in by_file.get(relpath, ()):
            last = n["qualname"].split(".")[-1]
            if last != func and n["qualname"] != func:
                continue
            if n["firstlineno"] <= lineno and (best is None or n["firstlineno"] > best["firstlineno"]):
                best = n
        if best is None:
            missing += 1
        else:
            found.add(best["i"])
    return found, missing


def stack_dist_scores(trace: dict, mapping: dict, ftests: list[str], max_depth: int = 50):
    sources, missing = fail_frame_nodes(trace, ftests)
    adj: dict[int, set] = defaultdict(set)
    for t in ftests:
        for e in trace["tests"][t].get("edges", []):
            a, b = e[0], e[1]
            if a is None or b is None:
                continue
            adj[a].add(b)
            adj[b].add(a)
    dist = bfs_levels(adj, sources, max_depth)
    elig = eligible_map(trace, mapping)
    scores: dict[str, float] = {}
    for i, d in dist.items():
        nid = elig.get(i)
        if nid is None:
            continue
        scores[nid] = max(scores.get(nid, -1e9), -float(d))
    return scores, {"n_fail_frames_found": len(sources), "n_fail_frames_missing": missing}


# ----------------------------------------------------------------- static methods
def build_name_index(graph: Graph) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for nid in graph.ids:
        if candidate_ok(nid):
            idx[graph.name.get(nid, nid.rsplit(".", 1)[-1])].append(nid)
    return idx


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def seeds_from_issue(text: str, graph: Graph, name_index: dict | None = None,
                     max_ambiguous: int = 3) -> set[str]:
    """Graph nodes named in the issue text.

    exact id match; dotted token matching an id suffix; plain token matching a
    short node name, kept only if the name is not ambiguous (<= max_ambiguous nodes).
    """
    name_index = name_index if name_index is not None else build_name_index(graph)
    seeds: set[str] = set()
    toks = set(_IDENT.findall(text or ""))
    suffix_hits: dict[str, list[str]] = {}
    dotted = [t for t in toks if "." in t]
    if dotted:
        for nid in graph.ids:
            if not candidate_ok(nid):
                continue
            for t in dotted:
                if nid == t or nid.endswith("." + t):
                    suffix_hits.setdefault(t, []).append(nid)
    for t in toks:
        if t in graph.node_set and candidate_ok(t):
            seeds.add(t)
    for hits in suffix_hits.values():
        if len(hits) <= max_ambiguous:
            seeds.update(hits)
    for t in toks:
        if "." in t or len(t) < 3:
            continue
        hits = name_index.get(t, [])
        if 0 < len(hits) <= max_ambiguous:
            seeds.update(hits)
    return seeds


def khop_scores(graph: Graph, seeds: set[str], k: int) -> dict[str, float]:
    dist = bfs_levels(graph.adj_all, seeds, k)
    return {n: -float(d) for n, d in dist.items() if candidate_ok(n)}


# ----------------------------------------------------------------- evaluation
def rank_eval(scores: dict[str, float], gold: set[str], ks=KS, n_shuffles: int = 100, seed: int = 0) -> dict:
    ids = list(scores)
    n = len(ids)
    res = {"n_candidates": n, "rank_of_best_gold": None, "hit": {k: 0.0 for k in ks}}
    gold_in = [g for g in gold if g in scores]
    if not n or not gold_in:
        return res
    rng = np.random.default_rng(seed)
    s = np.round(np.array([scores[i] for i in ids], dtype=float), 12)
    idx = {i: j for j, i in enumerate(ids)}
    rand = rng.random((n_shuffles, n))
    best = np.full(n_shuffles, np.inf)
    for g in gold_in:
        j = idx[g]
        better = int((s > s[j]).sum())
        ties = s == s[j]
        ties[j] = False
        before = (rand[:, ties] < rand[:, [j]]).sum(axis=1) if ties.any() else np.zeros(n_shuffles)
        best = np.minimum(best, better + before + 1)
    res["rank_of_best_gold"] = float(best.mean())
    res["hit"] = {k: float((best <= k).mean()) for k in ks}
    return res


def _seed_for(instance_id: str, method: str) -> int:
    return zlib.crc32(f"{instance_id}|{method}".encode()) & 0x7FFFFFFF


def rank_task(task: dict, trace: dict, graph: Graph, gold: set[str], bm25: BM25 | None = None,
              name_index: dict | None = None, ks=KS, n_shuffles: int = 100):
    """All methods for one task. Returns (ranking_rows, summary_dict, gold_in_trace_row)."""
    iid, repo = task["instance_id"], task["repo"]
    mapping = map_trace(trace, graph)
    ftests, ptests = failed_tests(trace), passed_tests(trace)
    gold = {g for g in gold if g in graph.node_set}

    methods: dict[str, dict[str, float]] = {}
    och, och_info = ochiai_scores(trace, mapping)
    methods["ochiai"] = och
    sd, sd_info = stack_dist_scores(trace, mapping, ftests)
    methods["stack_dist"] = sd
    nidx = name_index if name_index is not None else build_name_index(graph)
    seeds = seeds_from_issue(task.get("problem_statement", ""), graph, nidx)
    for k in (1, 2, 3):
        methods[f"static_k{k}"] = khop_scores(graph, seeds, k) if seeds else {}
    if bm25 is None:
        bm25 = BM25({n: tokenize(graph.text[n]) for n in graph.ids if candidate_ok(n)})
    methods["bm25"] = bm25.scores(tokenize(task.get("problem_statement", "")))

    rows = []
    for m in METHODS:
        r = rank_eval(methods[m], gold, ks, n_shuffles, _seed_for(iid, m))
        for k in ks:
            rows.append({
                "instance_id": iid, "repo": repo, "method": m, "K": k,
                "hit": r["hit"][k], "hit_bin": int(r["hit"][k] >= 0.5),
                "rank_of_best_gold": r["rank_of_best_gold"],
                "n_candidates": r["n_candidates"], "n_gold": len(gold),
            })

    # ---- gold in trace (oracle ceiling; uses the failing tests)
    tests = trace.get("tests", {})
    elig = eligible_map(trace, mapping)
    exec_f: set[str] = set()
    for t in ftests:
        exec_f |= mapped_ids_of_nodes(trace, mapping, tests[t].get("nodes", []), elig)
    exec_p: set[str] = set()
    for t in ptests:
        exec_p |= mapped_ids_of_nodes(trace, mapping, tests[t].get("nodes", []), elig)
    gold_exec = gold & exec_f
    lab, _ = label_edges(trace, graph, iid, repo, mapping=mapping, tests=ftests)
    n_static = n_dark = 0
    for r in lab:
        if r["phase"] != "runtime" or r["test_code"] or r["kind"] not in ("direct", "lifted"):
            continue
        if r["caller_id"] in gold_exec or r["callee_id"] in gold_exec:
            if r["static_edge"]:
                n_static += 1
            else:
                n_dark += 1
    frame_idx, _ = fail_frame_nodes(trace, ftests)
    frame_ids = mapped_ids_of_nodes(trace, mapping, frame_idx, elig)            # non-test frames only
    frame_ids_any = {mapping[i] for i in frame_idx if mapping.get(i) is not None}  # incl. test functions
    # The graph contains test functions with their own call edges, so a fair static
    # neighbourhood starts from the failing frame wherever it is, test code included.
    near = set(bfs_levels(graph.adj_all, frame_ids_any, 3)) if frame_ids_any else set()
    git = {
        "instance_id": iid, "repo": repo,
        "in_trace": int(bool(gold_exec)),
        "in_trace_diff": int(any(g not in exec_p for g in gold_exec)),
        "dark_only": int(bool(gold_exec) and n_dark > 0 and n_static == 0),
        "n_gold": len(gold), "n_gold_in_trace": len(gold_exec),
        "n_inc_static": n_static, "n_inc_dark": n_dark,
        "gold_in_static_k3_of_fail": int(bool(gold & near)),
    }
    summary = {
        "instance_id": iid, "repo": repo, "n_gold": len(gold), "n_seeds": len(seeds),
        "n_tests": len(tests), **och_info, **sd_info,
        "n_trace_nodes": len(trace["nodes"]),
        "n_fail_frames_src": len(frame_ids), "n_fail_frames_mapped": len(frame_ids_any),
        "mapped_nodes": sum(1 for v in mapping.values() if v is not None),
    }
    return rows, summary, git
