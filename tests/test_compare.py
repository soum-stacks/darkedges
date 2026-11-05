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
