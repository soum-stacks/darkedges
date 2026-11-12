# Dark Edges + TraceLens

Code for the Gemma 4 Developer Agent Paper Track entry **"Dark Edges: The Gemma 4 code graphs miss method calls, not functions, and filling them in does not fix localisation"**.

TraceLens is a small, standard-library runtime call tracer for Python 3.12+. It uses `sys.monitoring`, with a `sys.setprofile` fallback. Around it sits a pipeline that compares runtime call edges from failing tests with the competition's released static code graphs.

## Main results (strict set, 103 tasks; details in the paper and `paper/results/`)

| Finding | Number |
|---|---|
| Runtime call pairs with no graph edge ("dark") | fastapi 0.47, requests 0.52, rich 0.81 (pooled 72% of 14,059 pairs) |
| Dark fraction by callee | module-level functions 12-17%; methods 87-100% |
| Gold function executed by failing tests / within 3 static hops of the failure | 81-83% / 24-26% |
| Recall@20: Ochiai over tests (oracle) vs static 2-hop from issue symbols | 0.63 vs 0.26 (pre-registered H2, supported) |
| Non-oracle ladder (existing tests only), recall@20 | gold run by existing tests 52% (vs 83% failing tests); BM25 0.46, BM25 + existing tests 0.52, oracle 0.63 |
| Static AST repair (self-calls, attributes, constructors) | recovers 41% of dark pairs (precision proxy 0.87); localisation +0.01 |
| E4B agent, gold file touched: no graph / graph tools / graph + tracer | 0.72 / 0.70 / 0.60 (H4 not supported; tracer used in 28% of runs) |

## Layout

| Path | What | Runs on |
|---|---|---|
| `src/tracelens/` | Tracer, pytest plugin, CLI (`python -m tracelens.run`), summariser | Any Python 3.12+ |
| `skills/tracelens/` | Agent-facing tool (`trace_run.py`): ranked suspects from a traced repro or test | Agent sandbox |
| `src/darkedges/` | Environment builder, gold nodes, edge labelling, rankings, link prediction, statistics | CPU |
| `scripts/03_trace_tasks.ipynb` | Trace all tasks (pre and post) | Kaggle CPU notebook, competition attached |
| `scripts/04`-`10` | T2 analysis: dark edges, rankings, link prediction, stats, figures F1-F5 | Laptop or cluster |
| `scripts/08_agent_check.ipynb`, `11`, `12`, `08b` | T3 agent check (H4) | Kaggle L4 x4 (pinned Python 3.12 env); stats on a laptop |
| `scripts/13_graph_repair.py` | Static repair rules R1-R3 and their effect | Laptop |
| `scripts/14_nonoracle_ladder.py` | Non-oracle runtime ladder (settings pre-committed in `docs/PRECOMMIT_ladder.md`) | Laptop |
| `configs/protocol.yaml` | Pre-registered hypotheses and analysis plan (frozen at tag `protocol-v1`) | - |
| `docs/` | Amendments and choices (`t2_choices.md`, `t3_choices.md`), environment recipe, runbooks, completion reports | - |
| `paper/` | Figures, aggregate result tables, LaTeX source | - |
| `tests/` | Unit tests on synthetic data (`python -m pytest tests -q`) | - |

## Reproduce

1. Trace on Kaggle: attach the competition and run `scripts/03_trace_tasks.ipynb`. It writes `data/traces/`.
2. Get gold nodes and embeddings with `scripts/03b_gold_and_embeddings.py` (Kaggle CPU).
3. Analysis: `python scripts/04_compare_static.py`, then `05`, `06`, `07`, `08_callee_split.py`, `10`, `13_graph_repair.py` and `14_nonoracle_ladder.py`.
4. Agent check: `scripts/08_agent_check.ipynb` on Kaggle L4 x4, then `python scripts/12_agent_stats.py`.

Environment: `conda env create -f environment.yml`. Seed 0, 100 tie-break shuffles, 10,000 task-clustered bootstrap resamples.

## Data policy

Competition data (tasks, snapshots, graphs, embeddings) and per-task traces are **not** included, per the competition rules. Everything above regenerates from the competition data. Only code and aggregate tables are released.

## Tested

36+ unit tests on synthetic data. The tracer and pytest plugin were verified on real tasks with Python 3.13 (Kaggle CPU) and 3.12. The agent tool was verified inside the competition harness, called via `run_command`. Delivery through an ADK skill inside the Docker sandbox is not yet tested.

## License

Apache-2.0 (see `LICENSE`). Install the tracer only: `pip install git+https://github.com/soum-stacks/darkedges-paper`.
