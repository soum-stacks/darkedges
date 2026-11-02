## 2026-11-03 Phase 1 design decisions (append to docs/decisions.md)
- Test selection: run the WHOLE test files touched by test_patch (tasks.jsonl has no FAIL_TO_PASS list); fail-to-pass = failed pre AND passed post, computed in data/traces/_valid.jsonl. Whole files keep passing tests available for Ochiai (Section 8).
- Gold nodes are computed on Kaggle (needs snapshots; Helix has no snapshots) into results/gold_nodes.jsonl; extra optional field "unmatched" (additive).
- _env.jsonl (private log) added next to _status.jsonl: resolver notes per task.
- envbuild.mini_resolve intersects constraints per package (fixed point) instead of first-come-first-served; no backtracking; unsatisfiable/no-wheel reported in notes.
