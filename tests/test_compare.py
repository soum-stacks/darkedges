import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth  # noqa: E402
from darkedges.causes import classify  # noqa: E402
from darkedges.compare import (coverage_row, is_test_id, is_test_path, label_edges, map_node,  # noqa: E402
                               map_trace, task_dark_summary)
from darkedges.dataio import Graph  # noqa: E402


def _graph(prefix="fastapi"):
    g = synth.make_graph(prefix)
    return Graph(g["nodes"], g["edges"])


def test_map_node_layouts():
    ns = {"requests.api.get", "fastapi.X", "pkg.mod.K.m"}
    assert map_node("src/requests/api.py", "get", ns) == "requests.api.get"
    assert map_node("fastapi/__init__.py", "X", ns) == "fastapi.X"
    assert map_node("pkg/mod.py", "K.m", ns) == "pkg.mod.K.m"
    assert map_node("pkg/mod.py", "K.m.<locals>.f", ns) is None
    assert map_node("pkg/other.py", "nope", ns) is None


def test_test_path_helpers():
    assert is_test_path("tests/test_x.py") and is_test_path("pkg/conftest.py") and not is_test_path("pkg/core.py")
    assert is_test_id("tests.test_a.test_x") and not is_test_id("fastapi.routing.f")


def test_causes():
    assert classify("@property\ndef render(self):\n  pass", "direct") == "property"
    assert classify("def __init__(self):\n  pass", "direct") == "dunder"
    assert classify("@app.get('/')\ndef f():\n  pass", "direct") == "decorator"
    assert classify("def f():\n  pass", "lifted") == "framework_lifted"
    assert classify("async def f():\n  pass", "direct") == "async"
    assert classify("def f():\n  pass", "direct") == "other"


def test_label_edges_and_summary():
    g, tr = _graph(), synth.make_trace("fastapi")
    rows, st = label_edges(tr, g, "t1", "fastapi/fastapi")
    pairs = {(r["caller_id"], r["callee_id"], r["kind"]): r for r in rows}
    assert len(rows) == 7, sorted(pairs)
    assert pairs[("fastapi.core.run", "fastapi.core.helper", "direct")]["static_edge"] is True
    d = pairs[("fastapi.core.run", "fastapi.core.Widget.render", "direct")]
    assert d["static_edge"] is False and d["cause"] == "property" and d["static_path_len"] is None
    assert pairs[("fastapi.core.run", "fastapi.core.Widget.render", "direct")]["phase"] == "runtime"
    assert pairs[("fastapi.core.run", "fastapi.hooks.on_event", "lifted")]["cause"] == "framework_lifted"
    # near miss: run -> helper -> fmt is a 2-path, so run -> fmt would be len 2; check the BFS value directly
    assert pairs[("tests.test_fastapi.test_x", "fastapi.core.run", "direct")]["test_code"] is True
    assert st["n_entry"] == 1 and st["n_module"] == 1 and st["n_unmapped"] == 0  # nested edge never appears
    s = task_dark_summary(rows, "t1", "fastapi/fastapi")
    assert (s["n_pairs"], s["n_dark"]) == (6, 3) and abs(s["dark_fraction"] - 0.5) < 1e-9
    assert (s["n_pairs_direct"], s["n_dark_direct"]) == (5, 2)


def test_static_path_len_two():
    g = _graph()
    tr = synth.make_trace("fastapi")
    tr["edges"].append({"caller": 0, "callee": 2, "kind": "direct", "count": 1})  # run -> fmt (2 hops statically)
    rows, _ = label_edges(tr, g, "t1", "r")
    r = [x for x in rows if x["callee_id"] == "fastapi.util.fmt" and x["caller_id"] == "fastapi.core.run"][0]
    assert r["static_edge"] is False and r["static_path_len"] == 2


def test_coverage():
    g, tr = _graph(), synth.make_trace("fastapi")
    c = coverage_row(tr, g, map_trace(tr, g), "t1", "r")
    assert c["n_funcs"] == 6 and c["n_unmapped"] == 0 and c["mapped_call_fraction"] == 1.0
    assert (c["n_module_nodes"], c["n_nested_nodes"], c["n_test_nodes"]) == (1, 1, 1)


def test_graph_name_is_short_even_if_raw_name_is_qualified():
    g = Graph([{"id": "pkg.mod.K.__init__", "name": "pkg.mod.K.__init__", "text": "def __init__(self):\n  pass"}], [])
    assert g.name["pkg.mod.K.__init__"] == "__init__" and g.raw_name["pkg.mod.K.__init__"] == "pkg.mod.K.__init__"
    assert classify(g.text["pkg.mod.K.__init__"], "direct", name="pkg.mod.K.__init__") == "dunder"


def test_class_hub_labels():
    nodes = [{"id": i, "name": i, "text": t} for i, t in [
        ("pkg.mod.K", "class K:\n    pass"), ("pkg.mod.K.a", "def a(self):\n    pass"),
        ("pkg.mod.K.b", "def b(self):\n    pass"), ("pkg.mod.K.__init__", "def __init__(self):\n    pass"),
        ("pkg.mod.f", "def f():\n    pass")]]
    edges = [{"source": "pkg.mod.f", "target": "pkg.mod.K", "type": "calls"},
             {"source": "pkg.mod.K", "target": "pkg.mod.K.b", "type": "calls"}]
    g = Graph(nodes, edges)
    mk = lambda i, q, ln: {"i": i, "relpath": "pkg/mod.py", "qualname": q, "firstlineno": ln, "calls": 1}  # noqa: E731
    tr = {"nodes": [mk(0, "f", 1), mk(1, "K.a", 5), mk(2, "K.b", 9), mk(3, "K.__init__", 12)],
          "edges": [{"caller": 0, "callee": 2, "kind": "direct", "count": 1},    # f -> K.b : 2 hops via class K
                    {"caller": 1, "callee": 2, "kind": "direct", "count": 1},    # K.a -> K.b : same class
                    {"caller": 0, "callee": 3, "kind": "direct", "count": 1}],   # f -> K.__init__ : dunder, no path
          "tests": {}}
    rows, _ = label_edges(tr, g, "t", "r")
    by = {(r["caller_id"].split(".")[-1], r["callee_id"].split(".")[-1]): r for r in rows}
    fb = by[("f", "b")]
    assert fb["static_path_len"] == 2 and fb["static_path_len_noclass"] is None and fb["same_class"] is False
    assert by[("a", "b")]["same_class"] is True and by[("f", "__init__")]["callee_dunder"] is True
    s = task_dark_summary(rows, "t", "r")
    assert s["n_pairs"] == 3 and s["n_dark"] == 3
    assert s["n_pairs_excl_same_class"] == 2 and s["n_pairs_strict"] == 1 and s["dark_fraction_strict"] == 1.0
    assert (s["n_dark_len2"], s["n_dark_len2_noclass"], s["n_dark_same_class"], s["n_dark_dunder_callee"]) == (1, 0, 1, 1)
