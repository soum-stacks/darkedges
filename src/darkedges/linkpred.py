"""T2.3: can dark edges be predicted from static structure / released embeddings?

Positives : unique dark runtime pairs (phase runtime, non-test, kind direct|lifted,
            no static `calls` edge), at most `max_pos` per task (seeded subsample).
Negatives : per positive (u, v): `n_random` degree-matched random callees
            (same log2(static degree+1) bin as v) and `n_hard` same-module
            callees (module of v). Never a static call pair in either direction
            nor any runtime pair observed in that task.
Scorers   : emb_cos, name_jaccard, common_neighbors, adamic_adar, ppr,
            same_module, and logistic regression over the six (leave-one-repo-out).
Hits@10   : for up to `hits_pos` positives per task, rank v among 99 tail
            replacements (half same-module, rest degree-matched); average rank
            for ties.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .compare import is_test_id
from .dataio import Graph
from .rankings import split_identifier

_EMPTY = frozenset()
SCORERS = ["emb_cos", "name_jaccard", "common_neighbors", "adamic_adar", "ppr", "same_module", "same_class"]


def module_of(nid: str) -> str:
    """Module part of a dotted id: components before the first Capitalised one
    (a class), else all but the last component."""
    comps = nid.split(".")
    for i, c in enumerate(comps):
        if c[:1].isupper():
            return ".".join(comps[:i])
    return ".".join(comps[:-1])


# ----------------------------------------------------------------- embeddings
def load_embeddings(emb_dir: Path | None, instance_id: str) -> dict[str, np.ndarray] | None:
    """Read data/extras/embeddings/<stem>.npz (arrays `ids`, `emb`) via index.json."""
    if emb_dir is None or not Path(emb_dir).exists():
        return None
    emb_dir = Path(emb_dir)
    idx_path = emb_dir / "index.json"
    if not idx_path.exists():
        return None
    stem = json.loads(idx_path.read_text()).get(instance_id)
    if not stem or not (emb_dir / f"{stem}.npz").exists():
        return None
    z = np.load(emb_dir / f"{stem}.npz", allow_pickle=False)
    ids, mat = z["ids"], z["emb"].astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return {str(i): mat[k] for k, i in enumerate(ids)}


# ----------------------------------------------------------------- context
class StaticCtx:
    def __init__(self, graph: Graph, emb: dict | None = None, alpha: float = 0.15, iters: int = 20,
                 drop_class_hubs: bool = False):
        from scipy import sparse

        self.g = graph
        self.emb = emb
        self.drop_class_hubs = drop_class_hubs
        # structural adjacency used by common_neighbors / adamic_adar / ppr
        self.adj = graph.adj_calls_noclass() if drop_class_hubs else graph.adj_calls
        self.alpha, self.iters = alpha, iters
        self.index = {n: i for i, n in enumerate(graph.ids)}
        n = len(graph.ids)
        rows, cols, vals = [], [], []
        for u, nbrs in self.adj.items():
            if u not in self.index or not nbrs:
                continue
            w = 1.0 / len(nbrs)
            for v in nbrs:
                if v in self.index:
                    rows.append(self.index[u])
                    cols.append(self.index[v])
                    vals.append(w)
        self.PT = sparse.csr_matrix((vals, (cols, rows)), shape=(n, n))  # transpose of row-stochastic P
        self.deg = {nid: len(self.adj.get(nid, ())) for nid in graph.ids}
        self.universe = [nid for nid in graph.ids if not is_test_id(nid)]
        self.bins: dict[int, list[str]] = {}
        self.by_module: dict[str, list[str]] = {}
        for nid in self.universe:
            self.bins.setdefault(self._bin(nid), []).append(nid)
            self.by_module.setdefault(module_of(nid), []).append(nid)
        self._ppr: dict[str, np.ndarray] = {}
        self._tok: dict[str, set] = {}

    def _bin(self, nid: str) -> int:
        return int(math.log2(self.deg.get(nid, 0) + 1))

    def tokens(self, nid: str) -> set:
        t = self._tok.get(nid)
        if t is None:
            t = set(split_identifier(self.g.name.get(nid, nid.rsplit(".", 1)[-1])))
            self._tok[nid] = t
        return t

    def ppr_from(self, u: str) -> np.ndarray:
        p = self._ppr.get(u)
        if p is None:
            e = np.zeros(len(self.g.ids))
            e[self.index[u]] = 1.0
            p = e.copy()
            for _ in range(self.iters):
                p = self.alpha * e + (1 - self.alpha) * (self.PT @ p)
            if len(self._ppr) > 2000:
                self._ppr.clear()
            self._ppr[u] = p
        return p

    def features(self, u: str, v: str) -> np.ndarray:
        g = self.g
        if self.emb is not None and u in self.emb and v in self.emb:
            cos = float(self.emb[u] @ self.emb[v])
        else:
            cos = 0.0
        tu, tv = self.tokens(u), self.tokens(v)
        union = len(tu | tv)
        jac = (len(tu & tv) / union) if union else 0.0
        nu, nv = self.adj.get(u, _EMPTY), self.adj.get(v, _EMPTY)
        common = nu & nv if len(nu) < len(nv) else nv & nu
        aa = sum(1.0 / math.log(self.deg[w]) for w in common if self.deg.get(w, 0) > 1)
        ppr = float(self.ppr_from(u)[self.index[v]])
        same = 1.0 if module_of(u) == module_of(v) else 0.0
        return np.array([cos, jac, float(len(common)), aa, ppr, same, 1.0 if self.same_class(u, v) else 0.0],
                        dtype=np.float64)

    def same_class(self, u: str, v: str) -> bool:
        cu = self.g.class_of(u)
        return cu is not None and cu == self.g.class_of(v)


# ----------------------------------------------------------------- sampling
def dark_positives(rows: list[dict], exclude_same_class: bool = False
                   ) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    pos, seen_all = set(), set()
    for r in rows:
        pair = (r["caller_id"], r["callee_id"])
        seen_all.add(pair)
        if exclude_same_class and r.get("same_class"):
            continue
        if (r["phase"] == "runtime" and not r["test_code"] and r["kind"] in ("direct", "lifted")
                and not r["static_edge"]):
            pos.add(pair)
    return sorted(pos), seen_all


def _draw(rng, pool: list[str], u: str, v: str, ok, n: int, taken: set) -> list[str]:
    out = []
    if not pool:
        return out
    for _ in range(n * 8):
        if len(out) >= n:
            break
        w = pool[int(rng.integers(len(pool)))]
        if w == u or w == v or w in taken or not ok(u, w):
            continue
        out.append(w)
        taken.add(w)
    return out


def build_task_samples(ctx: StaticCtx, rows: list[dict], rng, max_pos: int = 300,
                       n_random: int = 3, n_hard: int = 2, hits_pos: int = 50, n_tail: int = 99,
                       exclude_same_class: bool = False):
    g = ctx.g
    positives, seen = dark_positives(rows, exclude_same_class)
    positives = [(u, v) for u, v in positives if u in ctx.index and v in ctx.index]
    if len(positives) > max_pos:
        sel = rng.choice(len(positives), size=max_pos, replace=False)
        positives = [positives[i] for i in sorted(sel)]

    def ok(u, w):
        if exclude_same_class and ctx.same_class(u, w):
            return False   # keep negatives comparable to positives when same-class pairs are excluded
        return not (g.has_call(u, w) or g.has_call(w, u) or (u, w) in seen)

    X, y, hit_groups = [], [], []
    for k, (u, v) in enumerate(positives):
        X.append(ctx.features(u, v))
        y.append(1)
        taken: set = set()
        negs = _draw(rng, ctx.bins.get(ctx._bin(v), []), u, v, ok, n_random, taken)
        negs += _draw(rng, ctx.by_module.get(module_of(v), []), u, v, ok, n_hard, taken)
        for w in negs:
            X.append(ctx.features(u, w))
            y.append(0)
        if k < hits_pos:
            t2: set = set()
            tails = _draw(rng, ctx.by_module.get(module_of(v), []), u, v, ok, n_tail // 2, t2)
            tails += _draw(rng, ctx.bins.get(ctx._bin(v), []), u, v, ok, n_tail - len(tails), t2)
            if len(tails) >= 9:
                hit_groups.append(np.vstack([ctx.features(u, v)] + [ctx.features(u, w) for w in tails]))
    return {
        "X": np.vstack(X) if X else np.zeros((0, len(SCORERS))),
        "y": np.array(y, dtype=int),
        "hits": hit_groups,
        "n_pos_available": len(dark_positives(rows, exclude_same_class)[0]),
    }


# ----------------------------------------------------------------- evaluation
def hits_at(groups: list[np.ndarray], scorer, k: int = 10) -> float | None:
    """Fraction of groups where the true item (row 0) has average rank <= k."""
    if not groups:
        return None
    hit = 0
    for G in groups:
        s = np.asarray(scorer(G))
        better = int((s[1:] > s[0]).sum())
        ties = int((s[1:] == s[0]).sum())
        rank = 1 + better + 0.5 * ties
        hit += rank <= k
    return hit / len(groups)


def evaluate_loo(per_task: list[dict], heldout_repos: list[str], use_emb: bool = True, seed: int = 0) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    out = []
    cols_all = list(range(len(SCORERS)))
    cols_noemb = [i for i in cols_all if SCORERS[i] != "emb_cos"]
    for held in heldout_repos:
        test = [t for t in per_task if t["repo"] == held and len(t["y"])]
        train = [t for t in per_task if t["repo"] != held and len(t["y"])]
        if not test:
            continue
        Xte = np.vstack([t["X"] for t in test])
        yte = np.concatenate([t["y"] for t in test])
        groups = [G for t in test for G in t["hits"]]
        if len(set(yte.tolist())) < 2:
            continue
        n_pos, n_neg = int(yte.sum()), int((1 - yte).sum())

        def add(name, score_te, scorer_for_groups):
            out.append({
                "heldout_repo": held, "scorer": name,
                "auc": float(roc_auc_score(yte, score_te)),
                "ap": float(average_precision_score(yte, score_te)),
                "hits10": hits_at(groups, scorer_for_groups),
                "n_pos": n_pos, "n_neg": n_neg,
            })

        for ci, name in enumerate(SCORERS):
            if name == "emb_cos" and not use_emb:
                continue
            add(name, Xte[:, ci], lambda G, ci=ci: G[:, ci])

        if train:
            Xtr = np.vstack([t["X"] for t in train])
            ytr = np.concatenate([t["y"] for t in train])
            if len(set(ytr.tolist())) == 2:
                variants = [("logreg", cols_all)] if use_emb else []
                variants.append(("logreg_no_emb", cols_noemb))
                for name, cols in variants:
                    sc = StandardScaler().fit(Xtr[:, cols])
                    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
                    clf.fit(sc.transform(Xtr[:, cols]), ytr)
                    add(name, clf.decision_function(sc.transform(Xte[:, cols])),
                        lambda G, cols=cols, sc=sc, clf=clf: clf.decision_function(sc.transform(G[:, cols])))
    return out


def loo_task_scores(per_task: list[dict], held: str, use_emb: bool = True, seed: int = 0) -> dict | None:
    """Same models as evaluate_loo, but returns the per-task score arrays of the held-out repo
    (for clustered bootstrap). Returns {"y": [arrays], "scores": {scorer: [arrays]}, "task_ids": [...]}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    test = [t for t in per_task if t["repo"] == held and len(t["y"])]
    train = [t for t in per_task if t["repo"] != held and len(t["y"])]
    if not test:
        return None
    out = {"y": [t["y"] for t in test], "task_ids": [t.get("instance_id") for t in test], "scores": {}}
    for ci, name in enumerate(SCORERS):
        if name == "emb_cos" and not use_emb:
            continue
        out["scores"][name] = [t["X"][:, ci] for t in test]
    if train:
        Xtr = np.vstack([t["X"] for t in train])
        ytr = np.concatenate([t["y"] for t in train])
        if len(set(ytr.tolist())) == 2:
            cols_all = list(range(len(SCORERS)))
            cols_noemb = [i for i in cols_all if SCORERS[i] != "emb_cos"]
            variants = ([("logreg", cols_all)] if use_emb else []) + [("logreg_no_emb", cols_noemb)]
            for name, cols in variants:
                sc = StandardScaler().fit(Xtr[:, cols])
                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
                clf.fit(sc.transform(Xtr[:, cols]), ytr)
                out["scores"][name] = [clf.decision_function(sc.transform(t["X"][:, cols])) for t in test]
    return out
