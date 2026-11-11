import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from darkedges import stats as S  # noqa: E402
from darkedges.dataio import Graph  # noqa: E402
from darkedges.linkpred import loo_task_scores  # noqa: E402

spec = importlib.util.spec_from_file_location("callee_split", ROOT / "scripts" / "08_callee_split.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def test_auc_bootstrap_ranks_a_better_scorer_higher():
    rng = np.random.default_rng(0)
    y_list, good, bad = [], [], []
    for _ in range(12):
        y = np.array([1] * 20 + [0] * 40)
        y_list.append(y)
        good.append(y + rng.normal(0, 0.5, len(y)))
        bad.append(rng.normal(0, 1, len(y)))
    r = S.cluster_bootstrap_auc(y_list, {"good": good, "bad": bad}, [("good", "bad")], B=200)
    assert r["auc"]["good"]["point"] > 0.8 and 0.35 < r["auc"]["bad"]["point"] < 0.65
    c = r["contrasts"]["good-bad"]
    assert c["diff"] > 0.2 and c["lo"] > 0 and c["p_diff_le0"] == 0.0 and r["n_tasks"] == 12


def test_loo_task_scores_shapes():
    rng = np.random.default_rng(1)

    def task(repo):
        y = np.array([1, 0, 0, 1, 0, 0])
        X = rng.normal(size=(6, 7)) + y[:, None]
        return {"repo": repo, "X": X, "y": y, "hits": [], "instance_id": repo}

    pt = [task("a"), task("a"), task("b"), task("b"), task("c")]
    ls = loo_task_scores(pt, "a", use_emb=True)
    assert len(ls["y"]) == 2 and set(ls["scores"]) >= {"emb_cos", "ppr", "logreg", "logreg_no_emb"}
    assert all(len(s) == 2 and len(s[0]) == 6 for s in ls["scores"].values())
    assert "emb_cos" not in loo_task_scores(pt, "a", use_emb=False)["scores"]


def test_callee_categories():
    nodes = [{"id": i, "name": i, "text": t} for i, t in [
        ("p.m.K", "class K:\n    pass"), ("p.m.K.a", "def a(self):\n    pass"), ("p.m.K.b", "def b(self):\n    pass"),
        ("p.m.K.__len__", "def __len__(self):\n    pass"), ("p.m.L", "class L:\n    pass"),
        ("p.m.L.c", "def c(self):\n    pass"), ("p.n.f", "def f():\n    pass")]]
    g = Graph(nodes, [])
    rows = [{"caller_id": u, "callee_id": v, "static_edge": False} for u, v in [
        ("p.m.K.a", "p.m.K.b"), ("p.m.K.a", "p.m.L.c"), ("p.m.K.a", "p.n.f"), ("p.m.K.a", "p.m.K.__len__")]]
    pairs = {(p["callee_method"], p["same_class"], p["dunder"], p["same_module"]) for p in cs.categorize(rows, g)}
    assert pairs == {(True, True, False, True), (True, False, False, True), (False, False, False, False),
                     (True, True, True, True)}
    sel = [p for p in cs.categorize(rows, g) if cs.CATS["callee: method, other class, non-dunder"](p)]
    assert len(sel) == 1
