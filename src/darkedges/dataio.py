"""Loaders for the Dark Edges pipeline (T2). Stdlib only.

Layout assumed (see docs/t2_runbook.md):
  data/slim/tasks.jsonl, data/slim/graphs/*.json   (from slim.zip)
  data/traces/<instance_id>.<pre|post>.json        (from phase1_outputs.zip)
  data/extras/gold_nodes.jsonl, data/extras/embeddings/  (from t2_extras.zip)
  results/ ...                                     (written by scripts 04-07)
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path

from .causes import is_class_text

FAIL_OUTCOMES = {"failed", "error"}


def repo_root() -> Path:
    env = os.environ.get("DARKEDGES_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def repo_short(repo: str) -> str:
    return repo.split("/")[-1].lower()


# ----------------------------------------------------------------- tasks
def find_tasks_file(slim_dir: Path) -> Path:
    p = slim_dir / "tasks.jsonl"
    if p.exists():
        return p
    hits = sorted(slim_dir.rglob("tasks.jsonl"))
    if not hits:
        raise FileNotFoundError(f"tasks.jsonl not found under {slim_dir}")
    return hits[0]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(slim_dir: Path) -> list[dict]:
    return read_jsonl(find_tasks_file(slim_dir))


# ----------------------------------------------------------------- graphs
class Graph:
    """Static code graph (NetworkX node-link JSON, multigraph)."""

    def __init__(self, nodes: list[dict], edges: list[dict]):
        self.ids: list[str] = []
        self.text: dict[str, str] = {}
        self.name: dict[str, str] = {}      # SHORT name = last dotted component of the id
        self.raw_name: dict[str, str | None] = {}   # the graph's own `name` field, unused by the analysis
        for n in nodes:
            nid = n["id"]
            self.ids.append(nid)
            self.text[nid] = n.get("text") or ""
            self.raw_name[nid] = n.get("name")
            self.name[nid] = nid.rsplit(".", 1)[-1]
        self.node_set = set(self.ids)
        self.calls_out: dict[str, set] = defaultdict(set)
        self.calls_in: dict[str, set] = defaultdict(set)
        self.adj_calls: dict[str, set] = defaultdict(set)  # undirected, calls only
        self.adj_all: dict[str, set] = defaultdict(set)    # undirected, any type
        self.any_pairs: set[tuple[str, str]] = set()       # directed, any type
        self.edge_types: Counter = Counter()
        self._class_nodes = None
        self._class_of: dict = {}
        self._noclass_out = None
        self._noclass_und = None
        for e in edges:
            u, v = e.get("source"), e.get("target")
            if u is None or v is None:
                continue
            t = str(e.get("type", "")).lower()
            self.edge_types[t] += 1
            self.any_pairs.add((u, v))
            self.adj_all[u].add(v)
            self.adj_all[v].add(u)
            if t == "calls":
                self.calls_out[u].add(v)
                self.calls_in[v].add(u)
                self.adj_calls[u].add(v)
                self.adj_calls[v].add(u)

    def has_call(self, u: str, v: str) -> bool:
        return v in self.calls_out.get(u, ())

    # ---- class-node helpers (class nodes aggregate calls and act as hubs in the released graphs)
    @property
    def class_nodes(self) -> set:
        if self._class_nodes is None:
            self._class_nodes = {n for n in self.ids if is_class_text(self.text[n])}
        return self._class_nodes

    def class_of(self, nid: str):
        """Nearest enclosing class node (longest proper dotted prefix that is a class node), or None."""
        if nid not in self._class_of:
            parts, c = nid.split("."), None
            for i in range(len(parts) - 1, 0, -1):
                p = ".".join(parts[:i])
                if p in self.class_nodes:
                    c = p
                    break
            self._class_of[nid] = c
        return self._class_of[nid]

    def calls_out_noclass(self) -> dict:
        """Directed calls adjacency with every class node removed (they cannot act as intermediates)."""
        if self._noclass_out is None:
            cn = self.class_nodes
            self._noclass_out = {u: {v for v in vs if v not in cn} for u, vs in self.calls_out.items() if u not in cn}
        return self._noclass_out

    def adj_calls_noclass(self) -> dict:
        """Undirected calls adjacency with every class node removed."""
        if self._noclass_und is None:
            cn = self.class_nodes
            self._noclass_und = {u: {v for v in vs if v not in cn} for u, vs in self.adj_calls.items() if u not in cn}
        return self._noclass_und


def load_graph(path: Path) -> Graph:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    edges = d.get("edges", d.get("links", []))
    return Graph(d.get("nodes", []), edges)


def find_graph(slim_dir: Path, task: dict) -> Path | None:
    iid = task["instance_id"]
    short = repo_short(task["repo"])
    base = task["base_commit"]
    cands = [
        slim_dir / "graphs" / f"{iid}.json",
        slim_dir / "graphs" / f"{short}_{base}.json",
        slim_dir / f"{iid}.json",
        slim_dir / f"{short}_{base}.json",
    ]
    for c in cands:
        if c.exists() and c.stat().st_size > 100:
            return c
    for c in slim_dir.rglob(f"*{base}*.json"):
        if c.stat().st_size > 100:
            return c
    return None


class GraphCache:
    """Small cache keyed by path (graphs are shared between tasks)."""

    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self._d: dict[Path, Graph] = {}

    def get(self, path: Path) -> Graph:
        if path not in self._d:
            if len(self._d) >= self.maxsize:
                self._d.pop(next(iter(self._d)))
            self._d[path] = load_graph(path)
        return self._d[path]


def bfs_levels(adj: dict, sources, max_depth: int) -> dict:
    """Multi-source BFS; returns {node: distance} for distance <= max_depth."""
    dist = {s: 0 for s in sources}
    dq = deque(dist)
    while dq:
        u = dq.popleft()
        d = dist[u]
        if d >= max_depth:
            continue
        for w in adj.get(u, ()):
            if w not in dist:
                dist[w] = d + 1
                dq.append(w)
    return dist


# ----------------------------------------------------------------- traces
def load_trace(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def failed_tests(trace: dict) -> list[str]:
    return [t for t, v in trace.get("tests", {}).items() if v.get("outcome") in FAIL_OUTCOMES]


def passed_tests(trace: dict) -> list[str]:
    return [t for t, v in trace.get("tests", {}).items() if v.get("outcome") == "passed"]


def select_tasks(tasks: list[dict], traces_dir: Path, task_set: str = "strict"):
    """Return [(task, pre_trace_path)] for tasks in the requested validity set.

    f2p    : pre trace exists and >=1 test failed             (114 expected)
    strict : f2p AND post trace has 0 failed and >=1 passed   (103 expected)
    all    : every task that has a pre trace file
    Validity is derived from the trace files themselves (not _status.jsonl).
    """
    out = []
    for t in tasks:
        iid = t["instance_id"]
        pre = traces_dir / f"{iid}.pre.json"
        if not pre.exists():
            continue
        if task_set == "all":
            out.append((t, pre))
            continue
        try:
            tr = load_trace(pre)
        except Exception:
            continue
        if not failed_tests(tr):
            continue
        if task_set == "strict":
            post = traces_dir / f"{iid}.post.json"
            if not post.exists():
                continue
            try:
                tp = load_trace(post)
            except Exception:
                continue
            if failed_tests(tp) or not passed_tests(tp):
                continue
        out.append((t, pre))
    return out
