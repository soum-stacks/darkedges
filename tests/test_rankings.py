import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth  # noqa: E402
from darkedges.dataio import Graph  # noqa: E402
from darkedges.rankings import BM25, rank_eval, rank_task, seeds_from_issue, tokenize  # noqa: E402


def _setup():
    g = synth.make_graph("fastapi")
    graph = Graph(g["nodes"], g["edges"])
    task = {"instance_id": "t1", "repo": "fastapi/fastapi",
            "problem_statement": "run fails: helper called from fastapi.core.run raises in fmt"}
    return graph, task, synth.make_trace("fastapi")


def test_tokenize_and_bm25():
    assert "get" in tokenize("getUserName") and "getusername" in tokenize("getUserName") and "user" in tokenize("get_user")
    bm = BM25({"a": tokenize("def helper(x): return fmt(x)"), "b": tokenize("def other(): pass")})
    sc = bm.scores(tokenize("helper fails"))
    assert "a" in sc and "b" not in sc


def test_rank_eval_ties_and_miss():
    scores = {f"n{i}": 1.0 for i in range(4)}
    r = rank_eval(scores, {"n0"}, ks=(1, 5), n_shuffles=2000, seed=1)
    assert abs(r["hit"][1] - 0.25) < 0.05 and r["hit"][5] == 1.0 and abs(r["rank_of_best_gold"] - 2.5) < 0.15
    m = rank_eval(scores, {"zzz"}, ks=(1,), n_shuffles=10)
    assert m["hit"][1] == 0.0 and m["rank_of_best_gold"] is None
    s = rank_eval({"a": 2.0, "b": 1.0}, {"b"}, ks=(1, 2), n_shuffles=10)
    assert s["hit"][1] == 0.0 and s["hit"][2] == 1.0 and s["rank_of_best_gold"] == 2.0


def test_seeds():
    graph, task, _ = _setup()
    assert seeds_from_issue(task["problem_statement"], graph) == {"fastapi.core.run", "fastapi.core.helper", "fastapi.util.fmt"}


def test_rank_task_methods():
    graph, task, trace = _setup()
    rows, summ, git = rank_task(task, trace, graph, {"fastapi.core.helper"}, n_shuffles=200)
    hit = {(r["method"], r["K"]): r["hit"] for r in rows}
    assert hit[("stack_dist", 5)] == 1.0            # failing frame is helper itself
    assert hit[("ochiai", 10)] == 1.0 and 0.3 < hit[("ochiai", 5)] < 0.95   # 3-way tie for ranks 4-6
    assert hit[("static_k1", 5)] == 1.0 and hit[("bm25", 10)] == 1.0
    assert summ["n_failed"] == 1 and summ["n_passed"] == 1 and summ["n_fail_frames_found"] == 2
    assert git["in_trace"] == 1 and git["in_trace_diff"] == 0 and git["dark_only"] == 0


def test_gold_dark_only():
    graph, task, trace = _setup()
    _, _, git = rank_task(task, trace, graph, {"fastapi.util.parse"}, n_shuffles=10)
    assert (git["in_trace"], git["in_trace_diff"], git["dark_only"]) == (1, 1, 1)
    _, _, git2 = rank_task(task, trace, graph, {"fastapi.util.unused"}, n_shuffles=10)
    assert git2["in_trace"] == 0 and git2["dark_only"] == 0


def test_static_neighbourhood_starts_from_test_code_frame():
    g = synth.make_graph("fastapi")
    g["edges"].append({"source": "tests.test_fastapi.test_x", "target": "fastapi.core.run", "type": "calls", "key": 0})
    graph = Graph(g["nodes"], g["edges"])
    task = {"instance_id": "t1", "repo": "fastapi/fastapi", "problem_statement": ""}
    trace = synth.make_trace("fastapi")
    trace["tests"]["tests/test_a.py::test_fail"]["fail_frames"] = [["tests/test_fastapi.py", 4, "test_x"]]  # only the test frame
    _, summ, git = rank_task(task, trace, graph, {"fastapi.core.helper"}, n_shuffles=10)
    assert git["gold_in_static_k3_of_fail"] == 1          # test_x -> run -> helper
    assert summ["n_fail_frames_src"] == 0 and summ["n_fail_frames_mapped"] == 1
