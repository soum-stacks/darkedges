# Decision log

Dated entries, newest last. Every number below was read from a notebook output. Items marked **TBD** have not been measured yet. Log timestamps in `measurements.jsonl` are UTC.

## 2026-10-30: Phase 0 status

| Task             | Status                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| T0.4, T0.5, T0.6 | Done                                                                                                                                                                                                         |
| T0.7             | **Open, not blocking until Phase 3 (6 Nov).** L4 x4 and E4B cold start are done. Still missing: E4B seconds per task, weekly GPU quota and charge rate. Finish it in the first GPU session of Phase 3. |
| T0.8             | Done by its acceptance test (requests_6644). Spikes on fastapi, rich and httpx are still open and should run before the 3 Nov gate.                                                                          |
| T0.9             | Done: sandbox image is Python 3.13, tracer tests pass on 3.13.15.                                                                                                                                            |
| T0.10, T0.11     | Drafted 2026-10-30, to be run and committed (see their entries once done).                                                                                                                                   |

### T0.4 slim extraction (Kaggle CPU notebook): done

- 129 tasks, 127 unique base commits. Graphs found per repo (counted per unique commit): fastapi 67, requests 12, rich 47, httpx 1. Empty or missing graphs: none.
- No graph has zero `calls` edges. Smallest graph (nodes) per repo: requests 711, rich 1920, fastapi 3238, httpx 740. Median `calls` edges per graph: requests 1828, rich 5624, fastapi 2470, httpx 1618.
- `slim.zip` created; T0.5 (copy to Helix `data/slim/`) reported done.

### T0.6 pytest on the repo (Kaggle CPU notebook, Python 3.13.15, pytest 8.4.2): done

- `tests/test_tracer.py`: 5 passed (both backends, mapping, CLI script).
- Plugin check on a demo repo (one passing, one failing test): both tests recorded with correct outcomes, failure frames `test_bad -> f -> g` correct, exit code 1, backend `monitoring`.
- Not tested: real repos with many tests, pytest-xdist, test timeouts, heavy-test overhead.
- Known issue to fix before T3.1: `summarize.py` prints a `<module> -> <module>` edge under "calls through framework frames". It is an import-phase edge (pytest importing the test module). Drop `<module>` edges from the agent-facing summary; `compare.py` keeps them as phase `import`.

### T0.7 L4 x4 gate (starter notebook): partly done

- Accelerators: a plain competition-attached notebook offered only GPU T4 x2 and TPU v5e-8. The **starter notebook copy** (Code tab, Copy & Edit) offers L4 x4. Use that copy for every GPU job.
- GPU notebook Python is 3.12.x (CPU notebook is 3.13.15).
- 12B reference run, starter unchanged, 2 FastAPI tasks, tp=4: both tasks stopped by the sample caps (1 minute, 10 tool calls), no patch. 61.8 s and 61.6 s, 10 and 5 tool calls. This is the cap, not a solve time.
- E4B (`gemma-4-e4b-it-qat-w4a16-ct`), tp=4, `gpu_memory_utilization=0.90`: **vLLM cold start 276.4 s**.
- GPU memory shown by `nvidia-smi` with the 12B loaded: about 20.7 GB of 23 GB on each GPU. That is vLLM's 0.90 reservation, not the model size.
- **TBD:** E4B seconds per task. The task cell has the `log(...)` line but has not been rerun with E4B.
- Session panel readings around the E4B vLLM start (session timer, not the weekly quota): before 1h18m of 12h and 5.6 GB of 57.6 GB disk; after 1h26m of 12h and 6.2 GB disk. The disk growth is probably the E4B weights being read or cached (not checked).
- **TBD:** weekly GPU quota before and after, and the L4 x4 charge rate. Nothing recorded so far is the weekly quota. The two `quota` lines in `measurements_gpu.jsonl` (22:33:00Z and 22:38:41Z) contain only placeholder text; ignore them.
- Sample `eval_config.yaml` defaults: `timeout_seconds: 60`, `max_tool_calls: 10`, `max_time_minutes: 1`, `max_turns: 50`. The file is labelled an optional participant configuration. For timing runs the working copy was raised to 10 minutes and 50 tool calls. **TBD:** check the competition Overview and Rules for any hard ceiling on these, and for the real command timeout (the evaluation code defaults to 300 s when the file omits it).

### Which log lines to trust (`logs/measurements_gpu.jsonl`, `logs/measurements_cpu.jsonl`)

- GPU: the only valid measurement is `vllm_start` for `gemma-4-e4b-it-qat-w4a16-ct`, 276.4 s (22:38:02Z). The two earlier `vllm_start` lines for the 12B model (0.0 s) are wrong: the server was already running. The `gpu_mem` lines are the 12B model's 0.90 reservation. There is no `task` line yet, so no E4B seconds per task exist.
- CPU: the two `T0.8` lines (pre and post for requests_6644) are valid.

### T0.8 environment spike, requests_6644 (Kaggle CPU notebook): done for requests

- New test `test_request_url_trims_leading_path_separators` FAILED at base under tracing (`'//v:h'` vs `'/v:h'`) and PASSED after the gold patch.
- Overhead 1.2x (pre) and 1.1x (post) on a 1.6 s run. Startup dominates, so this is a lower bound. The earlier 40x was a call-bound microbenchmark.
- Trace sizes: 102 nodes and 103 edges (pre), 101 nodes and 102 edges (post).
- No installs were needed: pytest and the dependencies came from the Kaggle image, and the repo came from the snapshot via `PYTHONPATH`.
- **TBD:** the same spike on `fastapi_14492`, `rich_3278`, `httpx_3672` (see `docs/env_recipe.md`). Gate for 3 Nov is at least 100 of 129 tasks with a valid pre trace.

### T0.9 Python versions

- Competition sandbox image (`docker/Dockerfile.sandbox`): `python:3.13-slim`. Tracer tests pass on 3.13.15 (this log) and on 3.12.3 (earlier build). Both backends exercised.

### Facts from the competition files that affect later tasks

1. **Graph node ids have two forms.** Neighbors of a `docs_src` node came back as `fastapi/fastapi;fastapi.exceptions.HTTPException` (an `owner/repo;` prefix) while other nodes have no prefix. `mapping.py` and `compare.py` must normalize both forms. Count the prefixed ids over a few full graphs before Phase 1.
2. **Embeddings look nearly collapsed.** The top-5 "similar" nodes to an unrelated tutorial route handler scored 0.9991 to 1.0000. For H3, also score cosine after centering the vectors, and report both.
3. **The sandbox venv has no pip.** The harness logs that `ensurepip` is unavailable, so skill scripts must be stdlib only. `sandbox/setup.py` resolves dependencies from `/wheels` and installs the package editable. Only its first 100 lines have been read.
4. **`/wheels` has 124 files** (8 tagged cp312, 8 tagged cp313, rest pure Python). It contains pytest 6.2.5, 8.3.4 and 9.1.1, anyio and sqlmodel. A name search found no pytest-httpbin, trustme, inline-snapshot or dirty-equals wheels, so tests needing them may not run in the sandbox.
5. **Sandbox image contents:** git, patch, protobuf-compiler, pigz; pip packages pytest, pytest-timeout==2.1.0, typer, pdm-backend, setuptools, wheel, poetry-core, hatchling, flit-core, editables; shim files `imp.py` and `telnetlib.py`; `TEST_TMPDIR=/tmp`; `WORKDIR /workspace`.
6. **`sample_submission/` has no `skills/` folder** (it holds `agent.yaml`, `configs/`, `eval_config.yaml`, `prompts/`, `sub_agents/`). The skill format for T3.1 must come from the `adk_submission` package.
7. **Privacy:** the Kaggle dataset that holds this repo also contains `data/slim/` graphs (competition data). Keep it Private.

### Open items

- T0.7: E4B seconds per task; weekly quota and charge rate; competition caps and command timeout.
- T0.8: spike on one task each from fastapi, rich, httpx.
- T0.10 (Helix check) and T0.11 (`configs/protocol.yaml`, commit before any Phase 2 analysis).
