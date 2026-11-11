# T3 Completion Report: Dark Edges + TraceLens (Phase 3, agent work + graph repair)

Prepared 12 Nov 2026 (01:15 IST) for the project owner. All numbers come from the Kaggle agent run of 11 Nov 2026 and the laptop runs of 12 Nov 2026, as recorded in the project chat. Competition-derived content stays private (Rule 4b). Read together with the Handoff Pack, Addenda 1-2, the T1 Completion Report, Addenda 3-4 and `docs/t3_choices.md`.

## 1. Verdict

**T3 is complete for the Paper Track.** H4 was tested, and the result is a pre-registered null. A post hoc static graph repair was added and run. The main-track items (T3.2 submission variants, T3.4 daily submissions and finals) are deferred to after the paper deadline; they are due 1-2 Dec.

| ID | Task | Acceptance test | Status | Evidence |
|---|---|---|---|---|
| T3.1 | TraceLens skill (`skills/tracelens/`) | Skill runs inside a harness task | Done (experiment form) | Self-test in the L4 notebook passed; in the agent run the tracer returned output to the agent in 9 of 32 recovered tracelens runs. Delivered by `run_command` from a host path, not by ADK skill loading (see section 8) |
| T3.2 | `submission_e4b/` variants P0-P4 | `validate_directory` passes | Not started (main track) | Deferred |
| T3.3 | `08_agent_check`: 30 tasks x 2 seeds x 3 conditions | `results/agent_check.jsonl` | Done, with partial data loss | 180 of 180 episodes ran; summary over all 180 kept; 92 per-run rows rebuilt (`results/agent_check_run1_partial.jsonl`) |
| T3.4 | `09_build_submission`, finals | Two E4B finals chosen | Not started (main track) | Deferred |
| Extra | `13_graph_repair.py` (exploratory) | Summary + figure | Done | `results/repair_summary.md`, `paper/figures/F7_repair.png` |
| Extra | `12_agent_stats.py` (H4 statistics) | Summary + figure | Done on the recovered rows | `results/agent_check_summary.md`, `paper/figures/F6_agent_check.png` |

## 2. Timeline (IST)

- 11 Nov, 16:00-19:00: T3 code written and unit-tested (skill, 08, 11, 12, 13, PBS, `tests/test_t3.py`); 30 tasks selected on Helix (`configs/agent_check_tasks.txt`).
- 19:50-20:20: environment problems on Kaggle. The current GPU image is Python 3.13, but the wheelhouse is cp312; its Python 3.12 has no torch; on 3.13 vLLM lacks msgspec, xgrammar, outlines_core and cbor2; internet cannot be on with L4 x4. **Fix: Copy & Edit the 31 Oct GPU notebook and pin its original environment (Python 3.12.13).**
- 20:25-20:40: litellm circular import (fixed by importing swegemma first); vLLM up in 313 s; smoke test passed with `trace_ok = 1`; gold-file parser bug found and fixed.
- 20:42-23:45: full run (1.4 h wall). The session ended before the output zip was downloaded.
- 12 Nov, 00:00-00:35: 92 rows recovered from the downloaded notebook; concurrency incident found; decided not to re-run (amendment T3-A1).
- 00:38: graph repair run on the laptop (190 s).

## 3. Inventory of files (repo `darkedges/`; where each runs in brackets)

| File | Why it exists | Result when run |
|---|---|---|
| `skills/tracelens/SKILL.md`, `skills/tracelens/scripts/trace_run.py` [Kaggle, agent] | Agent-facing wrapper around `tracelens.run`. Prints exit code, traceback tail and ranked SUSPECTS (failure stack first, then Ochiai when tests both pass and fail, then runtime distance), plus runtime calls into methods. Output at most 4,000 chars; 85 s limit | Self-test OK on Python 3.12.13 and 3.13; unit test passes |
| `scripts/11_select_agent_tasks.py` [Helix or laptop] | 30 tasks from `gold_in_trace.csv`, httpx excluded, proportional with at least 3 per repo, seed 0 | 16 fastapi, 11 rich, 3 requests -> `configs/agent_check_tasks.txt` |
| `scripts/08_agent_check.ipynb` (.py twin) [Kaggle L4 x4, pinned env] | H4 experiment. Builds one agent dir per (condition, seed); vLLM tp=4; Evaluator with 4 concurrent tasks; resumable JSONL; Python and GPU guard; gold-file check; thread guard | Ran 180 episodes in 1.4 h |
| `scripts/08b_rebuild_jsonl_from_log.py` [laptop] | Rebuilds `agent_check.jsonl` from the printed run lines (.txt or .ipynb) | 92 rows recovered |
| `scripts/12_agent_stats.py` [laptop] | H4: per-condition means (task-clustered bootstrap), paired differences, exact McNemar, tool use, exploratory splits, F6 | Run on the recovered rows |
| `scripts/13_graph_repair.py` [laptop] | Static repair rules R1-R3 on the released node text; dark-pair recovery, precision proxy, static k-hop recall before and after, F7 | 103 tasks in 190 s |
| `scripts/pbs/13_t3.pbs` [Helix] | Optional batch job for 13 and 12 | Not needed (laptop) |
| `scripts/pack_code_for_kaggle.sh` | Packs `src/`, `skills/` and the task list for the private Kaggle dataset | darkedges-code new version |
| `tests/test_t3.py` | 3 tests: tracer wrapper ranking, allocation, repair rules | Pass (container, Python 3.13) |
| `docs/t3_choices.md` | Fixed details of the agent check (committed before the run) and amendment T3-A1 | - |
| `docs/t3_t4_runbook.md` | Step-by-step commands; laptop-first rule | - |

Private, not committed: `results/agent_check_run1_partial.jsonl`, `results/repair_*.csv|json|md`, `results/edges_labeled.jsonl` and `gold_in_trace.csv` (copied from Helix).

## 4. Method summary

- **Agent check (H4).** Gemma 4 E4B (`gemma-4-e4b-it-qat-w4a16-ct`), competition harness (swegemma 0.2.11, subprocess sandbox), vLLM 0.19.1 tp=4 on 4 x L4. Budget as frozen: 10 min, 50 tool calls, 60 turns, 100 s per command. Sampling copied from the sample submission: temperature 0.2, top_p 0.95, thinking budget 4096; seed 0 or 1.
- **Conditions.** All three use one root agent with the same 5-step prompt (reproduce in `tests/_repro.py`, localise, fix, verify, submit); only step 2 differs. `no_graph_tools`: run_command, read_file, edit_file, write_file, get_status, submit_patch. `static_graph_tools`: plus get_code_neighbors, get_code_subgraph, search_similar_code. `tracelens`: plus an instruction to run the tracer on the reproduction first.
- **Primary outcome.** gold_file_touched: the agent patch modifies at least one non-test file that the gold patch modifies.
- **Graph repair.** R1 `self.m()` / `cls.m()` resolved in the class or its bases (up to 3 levels); R2 also `self.m` attribute loads; R3 also `X(...)` -> `X.__init__`. Edges are only added if absent. Scored on the T2 labelled runtime pairs (strict, headline filter) and by re-running static k-hop ranking with the same seeds, shuffles and K.

## 5. Results

### 5.1 Agent check (H4)

| | No graph tools | Static graph tools | TraceLens |
|---|---|---|---|
| Gold file touched, all 180 runs | 0.717 | 0.700 | 0.600 |
| Resolved, all 180 runs | 0.183 | 0.183 | 0.133 |
| Mean tool calls, all runs | 12.1 | 14.0 | 14.8 |
| Mean seconds per run, all runs | 165 | 206 | 204 |
| Gold file touched, 20 seed-0 tasks with all 3 conditions | 0.70 | 0.70 | 0.70 |

- H4 contrast on the recovered rows (tracelens - static, gold file touched): -0.067 [-0.267, +0.100], McNemar 2/2, p = 1. **H4 not supported.**
- Tracer output reached the agent in 9 of 32 recovered tracelens runs (28%). Gold-file rate 0.67 with tracer output, 0.65 without (exploratory).
- Graph tools did not beat no graph tools (0.70 vs 0.72, all runs).

### 5.2 Static graph repair (exploratory, strict 103 tasks)

| Repo | Original | R1 | R2 | R3 | Dark pairs recovered (R1/R2/R3) |
|---|---|---|---|---|---|
| fastapi (58) | 0.473 | 0.421 | 0.385 | 0.341 [0.321, 0.362] | 11% / 19% / 24% |
| rich (39) | 0.805 | 0.682 | 0.524 | 0.446 [0.417, 0.476] | 16% / 35% / 45% |
| requests (5) | 0.517 | 0.366 | 0.366 | 0.324 | 38% / 38% / 48% |

- Pooled: 10,067 dark pairs; recovered 1,546 / 3,272 / 4,160 (R3 = 41.3%). Same-class methods: 74% recovered (3,424 of 4,655). Other-class methods: 15% (736 of 4,825, constructors only). Module functions: 0 of 587.
- Precision proxy (added edges between two executed functions that were observed at run time): R1 0.871, R2 0.876, R3 0.863.
- Static k-hop recall@K on the repaired graph: +0.01 to +0.02, with CIs including 0 (k2@20: 0.259 -> 0.267, diff +0.007 [-0.011, +0.032]). Median candidate count k2 47 -> 51, k3 86 -> 109.

## 6. Interesting results and insights for the paper

**I1. A 4B agent mostly does not use a runtime tool, even when told to.**
- Data: 28% tracer uptake; the prompt asked for it as step 2.
- Caveats: recovered rows only; uptake is counted from tool outputs, not intent.
- Context: debugger uptake of 0.2-6.2% in arXiv:2610.36015; few chained graph queries in arXiv:2610.33958.

**I2. Graph tools gave E4B nothing over no graph tools.**
- Data: 0.70 vs 0.72 gold-file rate, and about 40 s more per run.
- Meaning: fits T2, where the graph lacks the method edges near the bug.
- Caveat: n = 30 tasks.

**I3. The released graphs can be partly repaired from their own text.**
- Data: 41% of dark pairs recovered with no execution, about 0.87 precision proxy.
- Meaning: the rest needs type inference (`self.attr.method()`) or runtime evidence.

**I4. Repair does not help localisation.**
- Data: class hubs already connect sibling methods within 2 hops.
- Meaning: the bottleneck is seeding (26 of 89 tasks have no issue symbol) and execution, not edge completeness. Ties T2 and T3 into one story.

**I5. Repo differences shrink after repair.**
- Data: rich-minus-fastapi drops from 0.33 to 0.10. Further support for the composition reading (method share) from T2.

**I6. Environment drift on Kaggle.**
- Data: the GPU image moved to Python 3.13 between 31 Oct and 11 Nov, and the wheelhouse is cp312-only.
- Meaning: a reproducibility note for the methods section; the main-track submission must be checked against the official image.

## 7. Problems found and fixed

| Problem | Cause | Fix |
|---|---|---|
| Wheel install failed (cp312 on Python 3.13) | Kaggle GPU image updated | Copy & Edit the 31 Oct notebook, pin the original environment |
| `litellm` circular import | Import order (litellm before swegemma) | Import swegemma first; restart kernel |
| `NameError: discover_adapters` | Import line deleted while editing | Restored the import |
| gold_file_touched always False | Gold patches use `---`/`+++` headers, not `diff --git` | New parser; assert that all 30 tasks have gold files before running |
| Two concurrent batches; per-run file lost | Cell re-run while the worker thread was alive; session ended before the download | Rebuilt 92 rows from the log; thread guard and warning added to the notebook; amendment T3-A1 |
| `git push` rejected; `paper/results` missing; tracked `.pyc` files | Remote ahead; folder absent; old commit tracked `__pycache__` | Commands in section 10 |

## 8. Deviations, amendments and limitations

- **Tool delivery.** The tracer was called via `run_command` from a host path, which works because the notebook harness uses the subprocess sandbox. A Docker submission must load it as an ADK skill; that path is untested.
- **T3-A1.** No re-run. H4 is reported from the complete-run means plus the recovered paired subset (20 tasks at seed 0). Only condition means were seen before this decision.
- **Graph repair.** Post hoc and exploratory; the precision proxy only sees executed pairs.
- **Small sample.** n = 30 tasks; one model, one prompt, one sampling setting. No Holm adjustment changes any conclusion (H2 p < 0.001; H1, H3 and H4 not supported).
- **Not tested.** Skill loading via ADK `skills:`; a forced-tracer condition (trace injected before the agent starts); failure-mode coding of transcripts (patches and logs were lost).

## 9. Hypotheses: final status

| Hypothesis | Status |
|---|---|
| H1 (fastapi most dark, requests least) | Not supported (rich > requests > fastapi) |
| H2 (trace ranking beats static 2-hop by 10 points at K=20) | Supported for the oracle methods (+0.37) |
| H3 (embeddings do not beat names) | Not supported (embeddings win by 0.08-0.15 AUC) |
| H4 (TraceLens raises the gold-file rate over static tools) | Not supported (null) |

## 10. Reproduction and next steps (T4)

1. Laptop, finish git (see the chat for the exact commands): create `paper/results`, untrack `__pycache__`, commit, `git pull --rebase`, `git push`.
2. Paper: Overleaf zip `darkedges_paper_overleaf.zip`. Copy F5 from Helix, check the numbers, submit the Kaggle Writeup before 12 Nov 23:59 UTC (13 Nov 05:29 IST).
3. Release: make the repo public after checking that no data is tracked; tag `paper-v1`. Optionally add a public data-free Kaggle notebook that reproduces the aggregate tables.
4. Main track (after the paper): pick the main-track submission design (skill delivery via ADK, forced tracer before the first turn), submissions P0-P4, two E4B finals by 1 Dec.
