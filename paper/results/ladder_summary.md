# Non-oracle runtime ladder (exploratory; docs/PRECOMMIT_ladder.md) - facts only

Tasks 89 (skipped 14); existing passing tests used 2652 of 3384 traced; tasks with no existing test 31; tasks where the issue retrieved >=1 test 56; k = 5.
Coverage ceiling (gold executed by any existing test): 0.517 [0.416, 0.618]
Gold inside the retrieved tests' functions: 0.449 [0.348, 0.551]

| Rung | R@1 | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|
| BM25 (issue text only) | 0.124 | 0.337 | 0.393 | 0.461 | 0.640 |
| Issue-retrieved existing tests | 0.035 | 0.091 | 0.126 | 0.171 | 0.275 |
| Hybrid: BM25 + existing tests | 0.090 | 0.281 | 0.404 | 0.517 | 0.674 |
| BM25 within tests' functions | 0.112 | 0.326 | 0.348 | 0.393 | 0.427 |
| Oracle: Ochiai with failing tests | 0.125 | 0.366 | 0.500 | 0.629 | 0.772 |

R@20 with CI: BM25 (issue text only) 0.461 [0.360, 0.562]; Issue-retrieved existing tests 0.171 [0.101, 0.249]; Hybrid: BM25 + existing tests 0.517 [0.416, 0.618]; BM25 within tests' functions 0.393 [0.292, 0.494]; Oracle: Ochiai with failing tests 0.629 [0.537, 0.719]

| Contrast | diff [95% CI] | McNemar a_only/b_only, p |
|---|---|---|
| R4_hybrid_rrf-R0_bm25@5 | -0.056 [-0.157, +0.045] | 8/13, p=0.383 |
| R4_hybrid_rrf-R0_bm25@20 | +0.056 [-0.022, +0.135] | 9/4, p=0.267 |
| R3_tests_idf-R0_bm25@5 | -0.246 [-0.357, -0.138] | 4/26, p=5.95e-05 |
| R3_tests_idf-R0_bm25@20 | -0.289 [-0.404, -0.179] | 5/30, p=2.24e-05 |
| R4b_bm25_in_tests-R0_bm25@5 | -0.011 [-0.135, +0.112] | 16/17, p=1 |
| R4b_bm25_in_tests-R0_bm25@20 | -0.067 [-0.191, +0.056] | 12/18, p=0.362 |
| R5_oracle_ochiai-R4_hybrid_rrf@5 | +0.085 [-0.033, +0.206] | 17/15, p=0.86 |
| R5_oracle_ochiai-R4_hybrid_rrf@20 | +0.112 [-0.023, +0.249] | 24/17, p=0.349 |

Gap closed at K=20, (R4 - R0) / (R5 - R0): 0.334 [-0.244, 1.303]

R@20 by repo: {"fastapi": {"n": 50, "R0_bm25": 0.32, "R3_tests_idf": 0.10060000000000001, "R4_hybrid_rrf": 0.38, "R4b_bm25_in_tests": 0.22, "R5_oracle_ochiai": 0.6468}, "rich": {"n": 33, "R0_bm25": 0.6060606060606061, "R3_tests_idf": 0.2487878787878788, "R4_hybrid_rrf": 0.6666666666666666, "R4b_bm25_in_tests": 0.6666666666666666, "R5_oracle_ochiai": 0.5872727272727273}, "requests": {"n": 5, "R0_bm25": 0.8, "R3_tests_idf": 0.4, "R4_hybrid_rrf": 0.8, "R4b_bm25_in_tests": 0.4, "R5_oracle_ochiai": 0.8539999999999999}}
