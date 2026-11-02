# Decision log

## 2026-10-31 (Day 0 complete)
- Protocol v1 frozen and tagged `protocol-v1` (configs/protocol.yaml).
- Environment recipe: mini-resolver, wheelhouse-first on PYTHONPATH (see docs/env_recipe.md). Strict pip resolution and the harness setup.py approach rejected (reasons there).
- Python: CPU notebook 3.13.x, GPU notebook 3.12.13. Tracer tests pass on 3.12.3 and 3.13.15; the spike ran on 3.13.
- Helix: Python 3.10.20, no scipy/networkx/sklearn/pandas/pyarrow; create a conda env with python=3.12 for analysis (internet works on compute nodes).
- Data facts: 129 tasks, 127 unique graphs (fastapi 67, rich 47, requests 12, httpx 1), none empty. httpx has n=1: no per-repo claims.
- GPU gate passed: 4 x NVIDIA L4 available in a notebook with the main competition attached.
  - E4B (gemma-4-e4b-it-qat-w4a16-ct, tp=4) vLLM start: 347.5 s (an earlier run: 276 s). GPU memory about 20.6 GiB per GPU at rest.
  - 3 tasks under a 10 min / 50 tool call budget: fastapi_14492 unresolved 48.7 s (5 calls), requests_6644 resolved 132.2 s (13 calls), rich_3278 unresolved 52.8 s (5 calls). n=3 is anecdotal. The agent stops far below the budget.
  - Quota: 30h available before, 29h 31m after; session wall 12.8 min (UI showed 15 min). Charge about 2x wall time. Planning figure: about 15 L4x4 hours per week.
  - Estimate (unverified): a full 129-task E4B run at about 78 s per task is about 2.8 h wall, about 5.7 quota hours. Use stratified subsets for development.
- Tracing overhead on short tests: 0.9x to 1.3x (earlier 40x was a call-bound microbenchmark, not representative).
