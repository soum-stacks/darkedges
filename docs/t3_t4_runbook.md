# T3 + T4 runbook (11-12 Nov 2026). Writeup deadline: 12 Nov 23:59 UTC = 13 Nov 05:29 IST

Paths: LAP repo = C:\Users\SOUMI\darkedges_core\darkedges ; HEL repo = /scratch/projects/soumi/gemma_competition/darkedges
PowerShell on the laptop, `ssh helix` alias works. All commands from the repo root unless stated.

## New files (all in the repo)
| File | Runs on | Purpose |
|---|---|---|
| skills/tracelens/SKILL.md, skills/tracelens/scripts/trace_run.py | Kaggle (agent) | TraceLens agent tool: trace a repro or tests, print ranked SUSPECTS + runtime method calls |
| scripts/11_select_agent_tasks.py | HEL | pick the 30 tasks -> configs/agent_check_tasks.txt |
| scripts/08_agent_check.ipynb (.py twin) | Kaggle L4x4 | H4 experiment: 30 tasks x 3 conditions x 2 seeds -> agent_check_outputs.zip |
| scripts/12_agent_stats.py | HEL | H4 statistics, results/agent_check_summary.md, paper/figures/F6 |
| scripts/13_graph_repair.py | HEL | static graph repair (exploratory), results/repair_summary.md, paper/figures/F7 |
| scripts/pbs/13_t3.pbs | HEL | runs 13, then 12 if agent_check.jsonl exists |
| scripts/pack_code_for_kaggle.sh | LAP | now packs src/ + skills/ + configs/agent_check_tasks.txt |
| tests/test_t3.py | LAP/HEL | 3 unit tests (tracer wrapper, allocation, repair rules) |
| docs/t3_choices.md | - | fixed details of the agent check; COMMIT BEFORE THE KAGGLE RUN |

## Step 1 (LAP, 5 min) commit and push
    git add skills scripts tests docs
    git commit -m "T3: TraceLens skill, agent check, graph repair, stats"
    git push

## Step 2 (HEL, 10 min) pull, test, select tasks, start graph repair
    ssh helix
    cd /scratch/projects/soumi/gemma_competition/darkedges && git pull
    conda activate /home/soumi/miniconda3/envs/darkedges
    python -m pytest tests/test_t3.py -q
    python scripts/11_select_agent_tasks.py          # prints TASK_IDS, writes configs/agent_check_tasks.txt
    cd scripts/pbs && qsub 13_t3.pbs && cd ../..     # graph repair, ~minutes; output logs/de_t3.o*
If `git pull` is not possible, copy with scp from the LAP repo root instead:
    scp scripts\11_select_agent_tasks.py scripts\12_agent_stats.py scripts\13_graph_repair.py scripts\pack_code_for_kaggle.sh helix:/scratch/projects/soumi/gemma_competition/darkedges/scripts/
    scp scripts\pbs\13_t3.pbs helix:/scratch/projects/soumi/gemma_competition/darkedges/scripts/pbs/
    scp tests\test_t3.py helix:/scratch/projects/soumi/gemma_competition/darkedges/tests/
    scp docs\t3_choices.md docs\t3_t4_runbook.md helix:/scratch/projects/soumi/gemma_competition/darkedges/docs/
    scp -r skills helix:/scratch/projects/soumi/gemma_competition/darkedges/
(If PBS says "bad interpreter" or "\r": `dos2unix scripts/pbs/13_t3.pbs`.)

## Step 3 (LAP, 5 min) bring the task list back, commit, pack
    scp helix:/scratch/projects/soumi/gemma_competition/darkedges/configs/agent_check_tasks.txt configs\
    git add configs/agent_check_tasks.txt && git commit -m "T3: agent-check task list (seed 0)" && git push
    bash scripts/pack_code_for_kaggle.sh            # Git Bash; writes darkedges-code.zip
Kaggle -> Datasets -> darkedges-code -> New Version -> upload darkedges-code.zip.

## Step 4 (Kaggle L4 x4, about 1.5-2 h wall, about 3-4 quota hours)
New notebook -> File -> Import notebook -> scripts/08_agent_check.ipynb.
Add Input: competition gemma-4-developer-agent; dataset metric/gemma-4-developer-agent-wheelhouse;
model google/gemma-4 gemma-4-e4b-it-qat-w4a16-ct (v2); dataset darkedges-code (latest version).
Accelerator GPU L4x4, Internet off. Check your GPU quota first (needs about 4 h left this week).
Run cells 1-6 interactively (setup, self-test, agent dirs, vLLM, runner, smoke). The smoke line must show
trace_ok >= 1. Then either run the remaining cells, or Save Version -> Save & Run All (background, resumable).
Download /kaggle/working/agent_check_outputs.zip from the Output panel. Paste back the summary table it prints.
If quota is short: set SEEDS = [0] (seed 1 can run after the deadline; report seed 0 as an interim result).

## Step 5 (LAP -> HEL, 10 min) H4 statistics
    ssh helix "mkdir -p /scratch/projects/soumi/gemma_competition/darkedges/results/agent_check_raw"
    scp $HOME\Downloads\agent_check_outputs.zip helix:/scratch/projects/soumi/gemma_competition/darkedges/results/agent_check_raw/
    ssh helix
    cd /scratch/projects/soumi/gemma_competition/darkedges/results/agent_check_raw && unzip -o agent_check_outputs.zip
    cp agent_check.jsonl ../ && cd ../.. && python scripts/12_agent_stats.py
    cat results/agent_check_summary.md results/repair_summary.md
Figures to the laptop for the Writeup:
    scp "helix:/scratch/projects/soumi/gemma_competition/darkedges/paper/figures/F*.png" paper\figures\

## Step 6 (T4) Writeup, release, submit
- Commit aggregates only: paper/figures/F6*, F7*, results summaries copied into paper/results/
  (agent_check_summary.md, repair_summary.md). Never commit results/*.jsonl/csv rows, agent patches or logs.
- Release: make the GitHub repo public only after `git ls-files | findstr /i "data results traces .zip"` shows
  nothing private; tag `paper-v1`.
- Writeup (your text, max 3,000 words incl. captions): abstract, introduction, methods, experiments, related
  work, citations. Attach figures; link the repo tag. Submit and save the confirmation.
