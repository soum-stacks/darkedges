#!/usr/bin/env python
"""Gate-3-Nov report: valid pre/post traces per repo (+ fail-to-pass test lists).
Run from the repo root on the laptop or Helix after unzipping phase1_outputs.zip:
    python scripts/03b_validity_report.py --traces data/traces --tasks data/slim/tasks.jsonl --gate 100
Writes data/traces/_valid.jsonl (private; not committed)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darkedges import tracing  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traces", default="data/traces"); ap.add_argument("--tasks", default="data/slim/tasks.jsonl")
ap.add_argument("--gate", type=int, default=100)
a = ap.parse_args()
tracing.validity_report(Path(a.traces).parent, a.tasks, gate=a.gate)
