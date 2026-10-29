# darkedges (core): tracelens tracer + Day-0 scripts

Stdlib-only runtime call-graph tracer (Python 3.12+ `sys.monitoring`, `sys.setprofile` fallback).

    PYTHONPATH=src python -m tracelens.run --root /workspace --out /tmp/trace.json -- pytest tests/test_x.py -q
    PYTHONPATH=src python -m tracelens.run --root /workspace --out /tmp/trace.json -- /tmp/repro.py
    PYTHONPATH=src python -m tracelens.summarize /tmp/trace.json 4500

Tested here (Python 3.12.3): edge logic on a synthetic repo (direct / lifted / entry edges, decorators, generators,
comprehensions, async, external framework frames) on BOTH backends, mapping, CLI + summarizer.
NOT tested here: the pytest plugin (pytest was not installed in this sandbox), real task environments, Python 3.13.
Run `pytest tests/` first on Kaggle/Helix.

Scripts: `scripts/01_extract_slim.py` (Kaggle: private slim zip + empty-graph report), `scripts/02_env_spike.py` (UNTESTED template).
