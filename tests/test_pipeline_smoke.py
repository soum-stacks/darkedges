"""Runs scripts 04 -> 07 on synthetic data (needs pandas, scipy, scikit-learn, matplotlib)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_pipeline(tmp_path):
    synth.build(tmp_path)
    root = ["--root", str(tmp_path)]
    out = _run("scripts/04_compare_static.py", *root)
    assert "dark_frac" in out and (tmp_path / "results" / "edges_labeled.jsonl").exists()
    _run("scripts/05_rankings.py", *root, "--shuffles", "20")
    assert (tmp_path / "results" / "rankings.csv").stat().st_size > 0
    _run("scripts/06_linkpred.py", *root)
    assert (tmp_path / "results" / "linkpred.csv").exists()
    _run("scripts/07_stats_figures.py", *root, "--boot", "200", "--min-tasks", "3")
    st = json.loads((tmp_path / "results" / "stats.json").read_text())
    assert "H1" in st and "H2" in st and "gold_in_trace" in st
    assert (tmp_path / "paper" / "figures" / "F1_dark_fraction.png").exists()
