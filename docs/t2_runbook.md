# T2 runbook (Phase 2, analysis)  -  written 5 Nov 2026

Order: 0 -> 1 -> 2 -> 3 (04 and 05 can run together) -> 4 -> 5 -> 6.
All repo paths are relative to the darkedges repo root. Competition-derived files stay private (Rule 4b).

| # | Platform | Run from | Command | Needs | Produces | Paste back |
|---|---|---|---|---|---|---|
| 0 | LAP | repo root | unzip t2_pack.zip over the repo; commit | - | code in place | - |
| 1 | KCPU (Kaggle notebook, competition attached, no accelerator) | one notebook cell | paste scripts/03b_gold_and_embeddings.py, run | competition data | /kaggle/working/t2_extras.zip (gold_nodes.jsonl, embeddings/) | the printed report |
| 1b | LAP -> HEL | - | download zip, scp to Helix, unzip into data/extras/ | step 1 | data/extras/gold_nodes.jsonl, data/extras/embeddings/ | - |
| 2 | HEL (interactive node) | repo root | conda env create -f environment.yml; conda activate darkedges; python scripts/00b_check_t2_inputs.py; python -m pytest tests -q | data/slim, data/traces | environment, preflight facts | preflight output, pytest summary line |
| 3 | HEL | scripts/pbs | qsub 04_compare.pbs ; qsub 05_rank.pbs | steps 1b, 2 | results/edges_labeled.jsonl, mapping_coverage.csv, dark_by_task.csv; rankings.csv, gold_in_trace.csv, task_summary.csv | last ~40 lines of each logs/de_compare.o*, logs/de_rank.o* |
| 4 | HEL | scripts/pbs | qsub 06_linkpred.pbs (after 04 finishes) | edges_labeled.jsonl | results/linkpred.csv | last ~40 lines of logs/de_linkpred.o* |
| 5 | HEL | scripts/pbs | qsub 07_stats.pbs (after 04-06) | all results | results/stats.json, hypotheses_summary.md, paper/figures/F1-F4 | cat results/hypotheses_summary.md |
| 6 | you | - | write findings in paper/notes.md | step 5 | - | - |

Before submitting: `mkdir -p logs results paper/figures` (PBS needs logs/ to exist).
Sensitivity (optional): rerun 04/05 with `--task-set f2p` into a copy of the repo root, or temporarily rename results/.
Debug on a few tasks interactively: `python scripts/04_compare_static.py --limit 5`.

## Files in this pack and where they go
src/darkedges/{dataio,causes,compare,rankings,linkpred,stats}.py
scripts/{00b_check_t2_inputs.py, 03b_gold_and_embeddings.py, 04_compare_static.py, 05_rankings.py, 06_linkpred.py, 07_stats_figures.py}
scripts/pbs/{04_compare,05_rank,06_linkpred,07_stats}.pbs
tests/{synth.py, test_compare.py, test_rankings.py, test_goldnodes.py, test_linkpred.py, test_stats.py, test_pipeline_smoke.py}
docs/{t2_runbook.md, t2_choices.md}      environment.yml (overwrite or merge with the existing one)
Not included on purpose: src/darkedges/__init__.py, hunks.py, envbuild.py (yours from T1; nothing here imports them).

## Tested / not tested
Tested here (synthetic data in the real formats, Python 3.12, numpy 2.5, scipy 1.18, sklearn 1.9, pandas 3.0, matplotlib 3.11): 26 unit/smoke tests, including scripts 04-07 end to end.
NOT tested: anything on real competition data (trace/graph field conventions below are assumptions), the Kaggle script 03b on real snapshots/embeddings, the PBS scripts under PBS Pro, runtime on full-size data (a 5,000-node synthetic graph runs in well under a minute per task).
Assumptions to confirm with step 2 / step 1 output: graph file location (data/slim/graphs/<id>.json or <repo>_<commit>.json), embedding file naming in the competition folder, whether graph node text keeps decorator lines, the order and format of `fail_frames`, that graph ids follow module.qualname.
