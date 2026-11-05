import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth  # noqa: E402
from darkedges.compare import label_edges  # noqa: E402
from darkedges.dataio import Graph  # noqa: E402
from darkedges.linkpred import (StaticCtx, build_task_samples, dark_positives, evaluate_loo,  # noqa: E402
                                load_embeddings, module_of)


def test_module_of():
    assert module_of("rich.console.Console.print") == "rich.console"
    assert module_of("fastapi.routing._prep") == "fastapi.routing"


def _ctx(prefix="fastapi", emb=None):
    g = synth.make_graph(prefix)
    graph = Graph(g["nodes"], g["edges"])
    rows, _ = label_edges(synth.make_trace(prefix), graph, "t", prefix)
    return StaticCtx(graph, emb), rows


def test_positives_and_features():
    ctx, rows = _ctx()
    pos, seen = dark_positives(rows)
    assert ("fastapi.core.run", "fastapi.core.Widget.render") in pos and len(pos) == 3
    f = ctx.features("fastapi.core.run", "fastapi.util.fmt")
    assert f.shape == (6,) and f[2] == 1.0 and f[5] == 0.0          # 1 common neighbour (helper); different module
    assert ctx.ppr_from("fastapi.core.run").sum() <= 1.0 + 1e-9


def test_samples_never_contain_forbidden_pairs():
    ctx, rows = _ctx()
    s = build_task_samples(ctx, rows, np.random.default_rng(0))
    assert s["X"].shape[0] == len(s["y"]) and s["y"].sum() == 3 and (s["y"] == 0).sum() > 0


def test_embedding_loader_and_loo(tmp_path):
    d = tmp_path / "emb"
    d.mkdir()
    ids = np.array(["fastapi.core.run", "fastapi.core.helper"])
    np.savez(d / "x.npz", ids=ids, emb=np.eye(2, 4, dtype=np.float16))
    (d / "index.json").write_text('{"t": "x"}')
    e = load_embeddings(d, "t")
    assert abs(float(e["fastapi.core.run"] @ e["fastapi.core.helper"])) < 1e-6
    assert load_embeddings(d, "missing") is None and load_embeddings(None, "t") is None

    per_task = []
    for repo in ("fastapi", "rich", "requests"):
        for k in range(3):
            ctx, rows = _ctx(repo)
            s = build_task_samples(ctx, rows, np.random.default_rng(k))
            s["repo"] = repo
            per_task.append(s)
    out = evaluate_loo(per_task, ["fastapi", "rich"], use_emb=False)
    scorers = {r["scorer"] for r in out}
    assert {"name_jaccard", "ppr", "logreg_no_emb"} <= scorers and "emb_cos" not in scorers
    assert all(0.0 <= r["auc"] <= 1.0 and r["n_pos"] > 0 for r in out)
