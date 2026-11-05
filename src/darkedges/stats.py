"""T2.4 statistics: task-clustered bootstrap (10,000 resamples), McNemar, Spearman."""
from __future__ import annotations

import numpy as np

B_DEFAULT = 10_000


def _ci(samples: np.ndarray, level: float = 0.95):
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def cluster_bootstrap_mean(values, B: int = B_DEFAULT, seed: int = 0) -> dict:
    """Mean over tasks with a percentile CI; tasks are the resampling unit."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = v[idx].mean(axis=1)
    lo, hi = _ci(means)
    return {"n": n, "mean": float(v.mean()), "lo": lo, "hi": hi}


def cluster_bootstrap_diff(a, b, B: int = B_DEFAULT, seed: int = 0) -> dict:
    """Paired per-task difference mean(a - b); a, b aligned by task."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[m]
    n = len(d)
    if n == 0:
        return {"n": 0, "diff": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = d[idx].mean(axis=1)
    lo, hi = _ci(means)
    return {"n": n, "diff": float(d.mean()), "lo": lo, "hi": hi}


def cluster_bootstrap_group_diff(x, y, B: int = B_DEFAULT, seed: int = 0) -> dict:
    """Difference of means between two independent task groups (x - y)."""
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    y = np.asarray([v for v in y if v is not None and not np.isnan(v)], dtype=float)
    if len(x) == 0 or len(y) == 0:
        return {"nx": len(x), "ny": len(y), "diff": None, "lo": None, "hi": None, "p_gt0": None}
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    by = y[rng.integers(0, len(y), size=(B, len(y)))].mean(axis=1)
    d = bx - by
    lo, hi = _ci(d)
    return {"nx": len(x), "ny": len(y), "diff": float(x.mean() - y.mean()), "lo": lo, "hi": hi,
            "p_gt0": float((d > 0).mean())}


def mcnemar_exact(a_bin, b_bin) -> dict:
    """Exact (binomial) McNemar test on paired binary outcomes."""
    from scipy.stats import binomtest

    a, b = np.asarray(a_bin, dtype=int), np.asarray(b_bin, dtype=int)
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"a_only": only_a, "b_only": only_b, "both": int(((a == 1) & (b == 1)).sum()),
            "neither": int(((a == 0) & (b == 0)).sum()), "p": p}


def spearman(x, y) -> dict:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or len(set(x[m].tolist())) < 2 or len(set(y[m].tolist())) < 2:
        return {"n": int(m.sum()), "rho": None, "p": None}
    r = spearmanr(x[m], y[m])
    return {"n": int(m.sum()), "rho": float(r.statistic), "p": float(r.pvalue)}
