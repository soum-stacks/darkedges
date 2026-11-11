# T3 choices and amendments (written 11 Nov 2026, BEFORE the agent-check run)

protocol.yaml stays frozen at tag protocol-v1. This file records the operational details that the protocol
left open, fixed before any agent episode was run (commit this file first; its commit time is the record).

## Agent check (H4, protocol agent_check)
- Model gemma-4-e4b-it-qat-w4a16-ct; budget exactly as frozen: timeout_seconds 100, max_tool_calls 50,
  max_time_minutes 10, max_turns 60; Phase 2 verification timeout 300 s (harness default).
- Tasks: 30 from the 89 ranked strict tasks (results/gold_in_trace.csv), httpx excluded, proportional by repo
  with at least 3 per repo, seed 0 (scripts/11_select_agent_tasks.py -> configs/agent_check_tasks.txt).
- Seeds 0 and 1, passed as generate_content_config.seed; other sampling parameters copied from the starter
  sample_submission/agent.yaml (recorded in run_meta.json).
- One root LlmAgent per condition, no sub-agents. Same instruction in all conditions except step 2 (localise):
  no_graph_tools = run_command, read_file, edit_file, write_file, get_status, submit_patch;
  static_graph_tools = those plus get_code_neighbors, get_code_subgraph, search_similar_code;
  tracelens = static_graph_tools plus the instruction to run the TraceLens tracer on the reproduction.
- Tool delivery in the experiment: the tracer (skills/tracelens/scripts/trace_run.py + src/tracelens) is placed
  on a host path and called with run_command. This works because the notebook harness uses the subprocess
  sandbox (shared file system). It is NOT how a Docker submission would see it; the competition submission
  must deliver it as an ADK skill (skills/tracelens). The H4 result therefore measures the value of the tool's
  output, not of the skill-loading mechanism.
- Harness: swegemma Evaluator, sandbox=subprocess, 4 tasks in parallel within a condition, seed-major order.
- Primary outcome O4 gold_file_touched: the agent patch modifies at least one non-test file that the gold patch
  modifies. Secondary: resolved (harness Phase 2). Paired by (task, seed); exact McNemar; task-clustered
  bootstrap (10,000) of per-task seed-averaged differences (scripts/12_agent_stats.py).
- Approximate transcript counts (string matches in the run trace/log; not used for H4): trace_ok (tracer output
  lines), trace_mentions, graph_mentions, error_responses, malformed_calls.
- Runs that raise an exception are retried once on re-run; still failing runs are reported as missing.
- Holm correction across H1-H4 as in the protocol.

## Exploratory, post hoc (decided after T2 results; label as such in the Writeup)
- scripts/13_graph_repair.py: static repair rules R1 (self/cls method calls, incl. base classes), R2 (+ self
  attribute references), R3 (+ constructor calls to __init__), applied to the released graph text only.
  Measures the share of T2 dark pairs recovered, a precision proxy, and static k-hop recall@K on the repaired
  graph (same seeds, tie shuffles and K as T2).
- Agent-check splits by repo and by T2 dark_only gold.
