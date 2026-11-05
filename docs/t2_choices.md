# T2 operationalisation choices (written 5 Nov 2026, BEFORE running any T2 script on real data)

The handoff pack fixes the definitions of dark edge, phase, gold node, the six ranking methods and the hypotheses.
Where it leaves a detail open, the T2 code makes the choices below. Append a pointer to this file in docs/amendments.md
(do not edit protocol.yaml). If any choice contradicts configs/protocol.yaml, the protocol wins: change the code, not the file.

## Task set
- Default `strict` = pre trace has >=1 failed test AND post trace has 0 failed and >=1 passed (expected 103). `f2p` (>=1 failed pre, expected 114) is a sensitivity run: `--task-set f2p`.
- Validity is derived from the trace files themselves, not from `_status.jsonl`.
- Headline per-repo claims only for repos with >=5 tasks (fastapi, rich, requests). httpx (1 task) is pooled in 'all' and used for training in link prediction only.

## Mapping and labelling (04, compare.py)
- Trace (relpath, qualname) -> graph id by suffix matching of the dotted module path (handles `src/` layouts and `__init__.py`); nested/lambda/comprehension frames never map (by design).
- `<module>` nodes: mapped only if the graph has a module node; otherwise their edges are counted (`edges_n_module` in mapping_coverage.csv) and excluded from labelled rows.
- Rows exist only for edges whose two endpoints both map. Self edges and entry edges are skipped (counted).
- `static_edge` = a graph edge of type `calls` caller->callee. `static_path_len` = directed BFS distance over `calls` edges, <=3, else null. `any_type_edge` records whether any other edge type links the pair.
- `phase` = import if either endpoint is `<module>` or its graph text is a `class` definition, else runtime.
- `test_code` = either endpoint's file is a test file (tests/, test/, test_*.py, *_test.py, conftest.py). Headline dark fractions exclude test_code rows.
- Headline dark fraction = per-task fraction of unique (caller, callee) pairs with kind direct|lifted, phase runtime, test_code false; reported as the task-macro mean. Sensitivities: direct-only pairs, count-weighted, import phase.
- Cause precedence: property > dunder > decorator > framework_lifted > async > other. `decorator` needs the graph node text to keep decorator lines; 00b prints how many nodes do.
- mapped_call_fraction = call-weighted share of mapped source nodes (module, nested and test nodes excluded from the denominator).

## Rankings (05, rankings.py)
- Candidates: graph nodes that are not test code (a dotted-id component is tests/test/conftest/test_*). A method only ranks nodes it has evidence for; unranked gold = miss.
- Ties: random, averaged over 100 shuffles (seed from instance id + method). `hit` is therefore in [0,1]; `hit_bin` = hit>=0.5 feeds McNemar.
- Ochiai: failing = outcome failed|error, passing = outcome passed, over the tests present in the pre trace; only functions executed by >=1 failing test are scored.
- stack_dist: undirected multi-source BFS over the union of the failing tests' runtime edges, sources = trace nodes matched to traceback frames (relpath + function name + nearest firstlineno <= lineno). Frames in test files are valid sources but are never candidates.
- static_k: seeds = graph nodes named in problem_statement (exact id, dotted id suffix, or a short name that matches <=3 non-test nodes); BFS over edges of ANY type, undirected (what get_code_neighbors exposes); candidates = nodes within k hops, ranked by distance.
- BM25: k1=1.5, b=0.75 over node text; query = problem_statement; identifiers split on snake/camel case and also kept whole.
- Gold-in-trace: in_trace = a gold node is executed by a failing test; in_trace_diff = ...and by no passing test; dark_only = in_trace and every incident runtime edge of the executed gold nodes (phase runtime, non-test, direct|lifted) is dark and there is at least one; also reported: gold within 3 static hops (any edge type) of a traceback-frame node.
- Tasks whose gold nodes are absent from the graph are skipped and listed in results/rankings_skipped.txt.

## Link prediction (06, linkpred.py)
- Positives: unique dark runtime pairs (same filter as the headline), <=300 per task (seeded subsample).
- Negatives per positive: 3 degree-matched (log2 static degree bin of the callee) + 2 same-module-as-callee; never a static call pair (either direction) or a runtime pair observed in the task.
- module_of(id) = components before the first Capitalised one, else all but the last (a heuristic: the graph has no file field).
- ppr = personalised PageRank (alpha 0.15, 20 iterations) from the caller over the undirected static `calls` graph.
- Hits@10: <=50 positives per task, rank of the true callee among 99 tail replacements (half same-module), average rank on ties.
- Leave-one-repo-out; logreg = standardised logistic regression (C=1) on the six scorers; logreg_no_emb drops emb_cos.
- Embeddings are the released 256-d vectors; nodes without a vector get cosine 0.0.

## Statistics (07, stats.py)
- Task-clustered percentile bootstrap, 10,000 resamples, seed 0; paired differences resample tasks jointly; exact McNemar (binomial) on hit_bin.
- Spearman at task level: dark_fraction vs dark_only (all tasks, and tasks with in_trace=1).
- H2 is evaluated for ochiai and stack_dist against static_k2 at K=20 (>=10 points and CI excludes 0 and McNemar p<0.05 are reported separately, not combined into a verdict).
- H3: per held-out repo, AUC(emb_cos) - AUC(name_jaccard); flagged if > 0.02.
- H4 (agent) is Phase 3, not computed here.
