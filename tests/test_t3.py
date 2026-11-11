"""T3 unit tests (synthetic data only): TraceLens agent wrapper, task allocation, static graph repair."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_trace_run_ranks_method_on_failure_path(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "core.py").write_text(
        "class T:\n    def run(self):\n        return self._w() + 1\n    def _w(self):\n        raise ValueError('bug')\n"
        "def unrelated():\n    return 1\n")
    (tmp_path / "tests" / "_repro.py").write_text("from pkg.core import T\nT().run()\n")
    out = subprocess.run([sys.executable, str(ROOT / "skills/tracelens/scripts/trace_run.py"), "tests/_repro.py"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=60).stdout
    assert "SUSPECTS" in out and "ValueError: bug" in out
    sus = out.split("SUSPECTS")[1]
    assert sus.index("T._w") < sus.index("T.run")          # innermost failure frame first
    assert "T.run -> T._w" in out                           # runtime method edge reported
    assert "unrelated" not in out                           # never executed


def test_allocate_min_three():
    m = _load("sel", "scripts/11_select_agent_tasks.py")
    assert m.allocate({"fastapi": 49, "rich": 33, "requests": 5}, 30) == {"requests": 3, "fastapi": 16, "rich": 11}
    assert sum(m.allocate({"fastapi": 10, "rich": 2}, 30).values()) == 12


def test_repair_rules():
    from darkedges.dataio import Graph
    m = _load("rep", "scripts/13_graph_repair.py")
    nodes = [{"id": "p.W", "text": "class W:\n    pass"},
             {"id": "p.W.__init__", "text": "def __init__(self):\n    pass"},
             {"id": "p.W.render", "text": "def render(self):\n    return 1"},
             {"id": "p.W.paint", "text": "def paint(self):\n    return self.render()"},
             {"id": "p.F", "text": "class F(W):\n    pass"},
             {"id": "p.F.go", "text": "def go(self):\n    x = W()\n    return self.render, self.paint()"},
             {"id": "p.f", "text": "def f():\n    return g()"}]
    g = Graph(nodes, [{"source": "p.W.paint", "target": "p.W", "type": "calls"}])
    add = m.repair_edges(g)
    assert add["R1"] == {("p.W.paint", "p.W.render"), ("p.F.go", "p.W.paint")}
    assert ("p.F.go", "p.W.render") in add["R2"] and ("p.F.go", "p.W.render") not in add["R1"]
    assert ("p.F.go", "p.W.__init__") in add["R3"]
    assert add["R1"] <= add["R2"] <= add["R3"]
