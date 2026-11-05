import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darkedges import stats as S  # noqa: E402


def test_bootstrap_mean():
    r = S.cluster_bootstrap_mean([0.1, 0.2, 0.3, 0.4], B=2000)
    assert abs(r["mean"] - 0.25) < 1e-9 and r["lo"] < 0.25 < r["hi"] and r["n"] == 4
    assert S.cluster_bootstrap_mean([])["mean"] is None


def test_diff_and_group_diff():
    d = S.cluster_bootstrap_diff([1, 1, 1, 0], [0, 0, 0, 0], B=1000)
    assert abs(d["diff"] - 0.75) < 1e-9
    g = S.cluster_bootstrap_group_diff([0.5, 0.6, 0.7], [0.1, 0.2, 0.15], B=1000)
    assert g["diff"] > 0.4 and g["lo"] > 0 and g["p_gt0"] == 1.0


def test_mcnemar_spearman():
    m = S.mcnemar_exact([1] * 10 + [1, 0], [0] * 10 + [1, 0])
    assert m["a_only"] == 10 and m["b_only"] == 0 and abs(m["p"] - 2 * 0.5 ** 10) < 1e-9
    assert S.mcnemar_exact([1, 0], [1, 0])["p"] == 1.0
    assert abs(S.spearman([1, 2, 3, 4], [2, 4, 6, 9])["rho"] - 1.0) < 1e-9
    assert S.spearman([1, 1, 1], [1, 2, 3])["rho"] is None
