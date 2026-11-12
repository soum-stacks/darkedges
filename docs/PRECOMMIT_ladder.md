# Pre-commitment: non-oracle runtime localisation ladder (exploratory)

Written 12 Nov 2026, 14:40 IST, BEFORE running scripts/14_nonoracle_ladder.py on real data. Commit this file first;
its commit time is the record. This analysis is exploratory (added after the pre-registered H1-H4).

Question: how much of the oracle runtime advantage (Ochiai over failing vs passing tests, recall@20 0.63 vs BM25 0.46)
survives when only evidence an agent could have BEFORE the fix is used?

Task set: strict, 89 ranked tasks (gold function in graph). Gold: data/extras/gold_nodes.jsonl.

Leakage filter ("existing tests"): a test in the pre trace is usable only if it PASSED in pre AND its file is not new in
the test patch AND its function name is not added (`+def test_...`) or named in a hunk header (`@@ ... def test_...`) of
the test patch. Failing tests are never used. Remaining leak, stated in the paper: the traced test FILES are those the
test patch touches (benchmark-selected pool).

Rungs (candidates = mapped, non-test graph functions):
- R0 BM25: issue text vs function source (as in T2).
- R1 coverage ceiling: share of tasks where any existing test executes a gold function.
- R2 issue -> test retrieval: BM25 of the issue against each existing test (nodeid tokens + test function source when it
  maps to a graph node); top k = 5 tests; candidate set = union of non-test functions they execute.
- R3 trace specificity: score(f) = sum over retrieved tests t of sim(issue, t) * idf(f), idf(f) = log(N / (1 + n_f)),
  N = number of existing tests in the task, n_f = existing tests executing f.
- R4 hybrid: reciprocal-rank fusion 1/(60 + rank_BM25) + 1/(60 + rank_R3).
- R4b: BM25 restricted to the R2 candidate set (sensitivity variant).
- R5 oracle Ochiai (as in T2, uses the failing tests).

Metrics: recall@K for K in 1, 5, 10, 20, 50 (random tie-breaks averaged over 100 shuffles), task-clustered bootstrap
10,000. Headline: gap closed at K = 20 = (R4 - R0) / (R5 - R0), with a bootstrap CI. Paired R4 vs R0: bootstrap diff +
exact McNemar on hit_bin. Per repo: fastapi and rich only (requests n = 5 reported but not interpreted).
All results reported whatever the direction.
