"""Synthetic mini-dataset in the real file formats (used by tests and by `python tests/synth.py <dir>`)."""
import json
import sys
from pathlib import Path

REPOS = ["fastapi/fastapi", "Textualize/rich", "psf/requests"]


def make_graph(prefix: str):
    ids = [f"{prefix}.core.run", f"{prefix}.core.helper", f"{prefix}.core.Widget", f"{prefix}.core.Widget.render",
           f"{prefix}.core.Widget.__init__", f"{prefix}.util.fmt", f"{prefix}.util.parse", f"{prefix}.util.unused",
           f"{prefix}.hooks.on_event", f"tests.test_{prefix}.test_x"]
    texts = {
        ids[0]: "def run(x):\n    return helper(x)",
        ids[1]: "def helper(x):\n    return fmt(x)",
        ids[2]: "class Widget:\n    pass",
        ids[3]: "@property\ndef render(self):\n    return fmt(self)",
        ids[4]: "def __init__(self):\n    pass",
        ids[5]: "def fmt(x):\n    return str(x)",
        ids[6]: "def parse(s):\n    return s",
        ids[7]: "def unused():\n    pass",
        ids[8]: "async def on_event(e):\n    pass",
        ids[9]: "def test_x():\n    run(1)",
    }
    nodes = [{"id": i, "name": i.rsplit(".", 1)[-1], "text": texts[i]} for i in ids]
    calls = [(0, 1), (1, 5), (3, 5)]
    edges = [{"source": ids[a], "target": ids[b], "type": "calls", "key": 0} for a, b in calls]
    edges.append({"source": ids[2], "target": ids[3], "type": "contains", "key": 0})
    return {"directed": True, "multigraph": True, "graph": {}, "nodes": nodes, "edges": edges}


def make_trace(prefix: str, fail: bool = True):
    f = lambda i, rel, q, ln: {"i": i, "relpath": rel, "qualname": q, "firstlineno": ln, "calls": 3}  # noqa: E731
    nodes = [
        f(0, f"{prefix}/core.py", "run", 10), f(1, f"{prefix}/core.py", "helper", 20),
        f(2, f"{prefix}/util.py", "fmt", 5), f(3, f"{prefix}/core.py", "Widget.render", 40),
        f(4, f"{prefix}/hooks.py", "on_event", 3), f(5, f"tests/test_{prefix}.py", "test_x", 1),
        f(6, f"{prefix}/core.py", "<module>", 1), f(7, f"{prefix}/core.py", "run.<locals>.inner", 12),
        f(8, f"{prefix}/util.py", "parse", 30),
    ]
    edges = [
        {"caller": 0, "callee": 1, "kind": "direct", "count": 2},     # static
        {"caller": 1, "callee": 2, "kind": "direct", "count": 2},     # static
        {"caller": 0, "callee": 3, "kind": "direct", "count": 1},     # dark (property)
        {"caller": 0, "callee": 4, "kind": "lifted", "count": 1},     # dark (dunder? no, async/lifted->decorator? lifted)
        {"caller": 5, "callee": 0, "kind": "direct", "count": 1},     # test edge (test code)
        {"caller": None, "callee": 5, "kind": "entry", "count": 1},
        {"caller": 6, "callee": 0, "kind": "direct", "count": 1},     # <module> edge
        {"caller": 0, "callee": 8, "kind": "direct", "count": 1},     # dark (other)
        {"caller": 3, "callee": 2, "kind": "direct", "count": 1},     # static
    ]
    t_fail = {"outcome": "failed" if fail else "passed", "nodes": [0, 1, 2, 3, 4, 5, 8],
              "edges": [[0, 1, "direct"], [1, 2, "direct"], [0, 3, "direct"], [0, 4, "lifted"],
                        [5, 0, "direct"], [0, 8, "direct"]],
              "fail_frames": [[f"tests/test_{prefix}.py", 4, "test_x"], [f"{prefix}/core.py", 22, "helper"]] if fail else []}
    t_pass = {"outcome": "passed", "nodes": [0, 1, 2, 5], "edges": [[0, 1, "direct"], [1, 2, "direct"], [5, 0, "direct"]],
              "fail_frames": []}
    return {"root": "/r", "python": "3.13", "backend": "monitoring", "wall_seconds": 1.0,
            "exit_code": 1 if fail else 0, "script_fail_frames": [], "nodes": nodes, "edges": edges,
            "tests": {"tests/test_a.py::test_fail": t_fail, "tests/test_a.py::test_ok": t_pass}}


def build(root: Path, n_per_repo: int = 6):
    slim, traces, extras = root / "data" / "slim", root / "data" / "traces", root / "data" / "extras"
    (slim / "graphs").mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    (extras).mkdir(parents=True, exist_ok=True)
    tasks, gold = [], []
    for r in REPOS:
        short = r.split("/")[-1].lower()
        for k in range(n_per_repo):
            iid = f"{short}_{100 + k}"
            base = f"{k:02d}" + "ab" * 19
            tasks.append({"instance_id": iid, "repo": r, "base_commit": base,
                          "problem_statement": f"run fails: helper called from {short}.core.run raises in fmt",
                          "patch": "", "test_patch": ""})
            g = make_graph(short)
            (slim / "graphs" / f"{short}_{base}.json").write_text(json.dumps(g))
            (traces / f"{iid}.pre.json").write_text(json.dumps(make_trace(short, True)))
            (traces / f"{iid}.post.json").write_text(json.dumps(make_trace(short, False)))
            gold.append({"instance_id": iid, "gold_node_ids": [f"{short}.core.helper"] if k % 3 else [f"{short}.core.Widget.render"],
                         "n_hunks": 1, "unmapped_hunks": 0})
    (slim / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    (extras / "gold_nodes.jsonl").write_text("\n".join(json.dumps(t) for t in gold) + "\n")
    return tasks


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/darkedges_synth")
    build(out)
    print("synthetic data in", out)
