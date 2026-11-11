# T2 analysis notes (working draft, 6 Nov 2026)

Working notes for you, NOT paper text. They hold the numbers, what each number does and does not support, and the
caveats. Write the Writeup in your own words (host guidance against LLM-written text; handoff rule 2).
All numbers come from results/hypotheses_summary.md and the step logs of the runs on 6 Nov 2026 unless marked PENDING.

## 0. What was run

- Tasks: 129 released; 103 in the primary ("strict") set = pre run has >=1 failing test AND post run (gold patch) has no failures
  and >=1 pass (fastapi 58, rich 39, requests 5, httpx 1). 26 tasks have no valid pair of runs (environment/collection failures).
- Ranking analyses use the 89 strict tasks that have >=1 gold function present in the graph (14 skipped). Gold functions come
  from the old-side lines of the gold patch: 112 of 129 tasks have >=1; 288 of 695 hunks (41%) map to no graph node (class-level or
  module-level edits, new files, async functions absent from the graphs).
- Software: Python 3.12.16, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1, matplotlib 3.11.2. Seed 0 throughout;
  10,000 task-clustered bootstrap resamples; 100 tie-break shuffles; exact McNemar.
- Choices not in the frozen protocol are listed in docs/t2_choices.md. Several (class-aware variants) were made AFTER seeing
  first results and are exploratory.

## 1. Facts about the released static graphs (these shape every result)

- Only `calls` edges exist (example graph: 4,256 nodes, 2,422 calls edges). 3,302 of its 4,256 nodes (78%) are test functions
  or live under tests. Tests are in the graph with their own call edges.
- No `async def` node exists in any of the four repos' graphs. Edges with an async endpoint cannot be labelled, so they are outside
  every dark-edge fraction. Call-weighted mapping coverage: rich, requests, httpx 1.000; fastapi 0.934 (unmapped examples:
  `FastAPI.__call__`, `AsyncExitStackMiddleware.__call__`).
- The graph's own `name` field equals the full dotted id for every node (4,256 of 4,256), not the short name.
- Decorator lines are almost never kept in node text (18 of 3,593 function-like nodes in the example graph), so the
  `decorator` and `property` cause labels cannot fire. Only `dunder`, `framework_lifted` and `other` are meaningful.
- Class nodes are hubs. In the example graph 2,184 of 5,460 calls edges (40%) start at a class node. Class nodes appear to
  stand in for calls made inside their methods (mechanism inferred from examples, not checked against the builder).
- Embeddings: 256-d, one per node, for all 127 unique graphs (151 MB as float16).

## 2. H1: dark-edge fraction by repo (task-macro mean, 95% task-clustered bootstrap CI)

Definition (protocol): runtime edge between two mapped repo functions, kind direct or lifted, with no `calls` edge caller->callee; import-phase
and test-code edges excluded; unit = unique (caller, callee) pair per task.

| repo (tasks) | headline | direct only | excl. same-class pairs | excl. same-class and dunder callees ("strict") |
|---|---|---|---|---|
| rich (39) | 0.805 [0.752, 0.843] | 0.805 [0.752, 0.843] | 0.678 [0.634, 0.712] | 0.554 [0.516, 0.588] |
| requests (5) | 0.517 [0.444, 0.591] | 0.459 [0.344, 0.573] | 0.411 [0.383, 0.440] | 0.287 [0.262, 0.312] |
| fastapi (58) | 0.473 [0.440, 0.509] | 0.410 [0.374, 0.451] | 0.421 [0.386, 0.460] | 0.249 [0.226, 0.269] |

Contrasts (difference of means, CI): rich - fastapi is -0.33 / -0.26 / -0.31 (headline / excl. same-class / strict) with every CI
excluding 0 (fastapi - rich shown in the summary file). fastapi - requests: -0.045 [-0.129, 0.033] headline, +0.009 [-0.038, 0.059] excl. same-class,
-0.038 [-0.071, -0.004] strict.

Reading:
- H1 predicted fastapi highest and requests lowest. Observed order is rich > requests > fastapi (headline), rich > fastapi > requests (excl. same-class),
  rich > requests > fastapi (strict). The prediction fails under every definition because rich is highest.
- Robust statement: rich is far above fastapi on all four definitions. Nothing reliable about requests (5 tasks; intervals from 5 tasks are
  too narrow; its rank flips with the definition).
- Rates depend strongly on how the graph represents classes: the strict rate is about half the headline for fastapi (0.249 vs 0.473).
- fastapi's rate is not comparable to rich's: edges with async endpoints are invisible (direction of bias unknown).
- Lifted edges (reached through framework frames): 428 dark vs 3 static of 431. Lifted edges are dark almost by construction. Only fastapi
  and requests have any (rich headline = direct-only).
- Pooled counts (pairs): 14,059 total, 10,067 dark; rich 9,799 pairs (70% of all) in 39 tasks (about 251 per task vs 71 for fastapi, 21
  for requests). Pooled percentages mostly describe rich; use task-macro numbers.

## 3. Composition of the 10,067 dark pairs

- Caller and callee in the same class: 4,655 (46.2%). Callee is a dunder method: 3,159 (31.4%). The two overlap (not additive; overlap not computed).
- Static path length of dark pairs, class nodes allowed as intermediates: 2 hops 6,712 (66.7%), 3 hops 738 (7.3%).
  With class nodes removed: 2 hops 528 (5.2%), 3 hops 277 (2.8%), no route within 3 hops 9,262 (92.0%).
  The near-miss rate is therefore almost entirely a class-node effect (1,936 of 2,000 sampled length-2 paths go through a class node).
- Dark-row causes (10,072 rows): dunder 3,164, other 6,599, framework_lifted 309.
- Interpretation to treat as a hypothesis: many dark edges are `self.method()` calls and constructor / protocol-method calls that the graph
  attributes to class nodes instead of methods. Whether they count as "missed" depends on what a class-level edge means.
  `HARNESS_README.md` in the competition folder may describe the builder: check before writing about mechanism.
- PENDING (extra 1, callee-type split): dark fraction for calls to module-level functions vs methods of other classes, per repo.
  The idea to test: rich's high rate comes from method calls across classes (calls on objects) which a name-based resolver cannot follow.

## 4. H2: localisation recall (89 tasks; task-macro mean [95% CI])

| method | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|
| ochiai (oracle) | 0.365 | 0.498 | 0.631 [0.539, 0.720] | 0.773 |
| stack_dist (oracle) | 0.421 | 0.484 | 0.573 [0.473, 0.673] | 0.684 |
| bm25 (issue text) | 0.337 | 0.393 | 0.461 [0.360, 0.562] | 0.640 |
| static_k2 (issue symbols, 2 hops) | 0.175 | 0.214 | 0.260 [0.176, 0.349] | 0.324 |

Paired contrasts at K=20 (diff [95% CI]; exact McNemar on hit>=0.5): ochiai - static_k2 +0.371 [0.256, 0.485] (42 vs 4 tasks, p<0.001);
stack_dist - static_k2 +0.313 [0.183, 0.441] (38 vs 9); ochiai - bm25 +0.170 [0.031, 0.306] (p=0.011); stack_dist - bm25 +0.112
[-0.034, 0.257] (p=0.243); ochiai - stack_dist +0.058 [-0.036, 0.151]; bm25 - static_k2 +0.201 [0.102, 0.302].

Reading:
- H2 (trace recall at K=20 beats static 2-hop by >=10 points) holds for both trace methods: +37 and +31 points, with lower CI bounds of
  25.6 (ochiai) and 18.3 (stack_dist) points.
- Caveat 1, oracle: ochiai and stack_dist use the failing tests that define the task; an agent would have to find or write a reproducer.
  They are ceilings. BM25 is the fair non-oracle comparison; ochiai leads it by 17 points, stack_dist by an unresolved 11 points.
- Caveat 2, static baseline coverage: 28 of 103 strict tasks (26 of the 89 ranked) have no issue-text seed, so static_k counts them as misses.
  On the 63 seeded tasks at K=20: ochiai 0.628, bm25 0.540, stack_dist 0.522, static_k2 0.367 (gaps 26 and 15 points, still above 10).
  This subset was chosen after seeing results. At K=5 BM25 (0.413) leads ochiai (0.361) and stack_dist (0.371): tracing helps most beyond the top few.
- Caveat 3: static baselines keep class hubs (the agent-visible neighbour tool exposes them). Hubs enlarge static reach, so this is conservative for H2.
- Caveat 4: 20 of 89 tasks have no passing test, so Ochiai ties heavily there (tie-averaged over 100 shuffles). Ranked tasks need a gold function in
  the graph; one task has 101 gold functions (easy recall). Only the two static_k2 contrasts at K=20 were pre-specified; the rest are
  exploratory and uncorrected. The script's "meets >=10 pts" flag looks at the point estimate only.

## 5. Where the gold edit is, relative to traces (89 tasks)

- Gold function executed by a failing test: 74 of 89 (0.831). Also not executed by any passing test: 26 (0.292). Reachable only through dark
  edges (every incident runtime edge is dark): 18 (0.202). Within 3 static hops (any edge type, class hubs included) of a traceback frame
  (test-code frames included): 21 (0.236).
- By repo: fastapi 0.820 / 0.440 / 0.140 / 0.220; rich 0.848 / 0.061 / 0.333 / 0.242; requests 1.000 / 0.400 / 0.000 / 0.200 (requests tiny).
- Statement the data supports: the edit location is run by the failing tests in about 8 of 10 tasks, but a static neighbourhood of the failing
  frame contains it in about 2 of 10.
- Spearman, task level, dark fraction vs gold-reachable-only-via-dark-edges: rho 0.253 (p=0.017, n=89); 0.327 (p=0.004, n=74 tasks with
  the gold executed). Both quantities come from the same edge labels, so part of this is mechanical, and rich (high on both) is a confounder. Do not
  present it as independent evidence.

## 6. H3: predicting dark edges (leave-one-repo-out, pooled AUC; no intervals yet)

Positives: unique dark pairs (<=300 per task, seeded). Negatives: 3 degree-matched + 2 same-module-of-callee per positive; never a static or observed pair.

Default variant (class hubs present): AUC (fastapi / requests / rich)
- emb_cos 0.703 / 0.734 / 0.695; name_jaccard 0.580 / 0.727 / 0.554; ppr 0.765 / 0.745 / 0.764; common neighbours 0.702 / 0.716 / 0.730;
  same_class 0.567 / 0.664 / 0.675; logreg 0.782 / 0.845 / 0.778; logreg without embeddings 0.791 / 0.857 / 0.792.
No-hubs variant (class nodes removed from the structural graph):
- common neighbours, Adamic-Adar, ppr fall to 0.51-0.57 (about chance); emb_cos 0.711 / 0.699 / 0.700; logreg 0.751 / 0.808 / 0.759; without embeddings 0.695 / 0.799 / 0.719.
Cross-class variant (no hubs, same-class pairs excluded from positives and negatives):
- emb_cos 0.697 / 0.613 / 0.629; name_jaccard 0.601 / 0.620 / 0.554; same_module 0.618 / 0.643 / 0.541; logreg 0.735 / 0.706 / 0.681;
  without embeddings 0.666 / 0.697 / 0.609; same_class 0.500 by construction. Positives: 1,438 / 35 / 3,924.

Reading:
- The structural scorers' apparent strength in the default variant is a class-hub artifact: a class node is a common neighbour of two methods of
  one class, and 46% of positives are same-class pairs.
- Embeddings beat name-token Jaccard by more than 0.02 AUC in 2 of 3 held-out repos in all three variants (fastapi +0.10 to +0.13, rich +0.08 to
  +0.15; requests about 0). H3 predicted they would not, so H3 is not supported as stated. Effect sizes are modest: AUC 0.63-0.71 is
  signal, not a good predictor.
- Embeddings add to the other scorers only once hubs are gone (logreg minus logreg-without-embeddings: -0.01 in the default variant, +0.04 to +0.07 on
  fastapi and rich in the other two). Hits@10 is mixed: name Jaccard beats embeddings on requests, and on rich in the cross-class variant.
- Limits: pooled AUCs, same pairs recur across tasks that share code, requests has 58 / 35 positives. PENDING (extra 2): clustered intervals for
  emb - name, emb - same_module, emb - ppr and logreg - logreg-without-embeddings.

## 7. PENDING: sensitivity on the f2p task set (extra 3)

120 tasks have >=1 failing pre test (vs 103 strict). Compare against the strict numbers above: H1 ordering and rich-minus-fastapi gap, H2
contrasts at K=20, in-trace and static-reach fractions, H3 emb vs name gaps. Report any difference larger than the strict intervals.

## 8. Claim checklist

Supported by this data (state with the stated conditions):
1. In the released graphs, a large share of runtime function-to-function edges between mapped repo functions has no `calls` edge; the share depends strongly
   on how classes are represented (headline 0.47-0.81, strict 0.25-0.55).
2. Rich has a much higher dark fraction than fastapi under every definition tried.
3. Failing-test-based ranking recovers the gold function far more often than a static neighbourhood from issue symbols (K=20: +31 to +37 points) and than BM25 (Ochiai, +17 points).
4. The gold edit is executed by the failing tests in 83% of tasks vs 24% within 3 static hops of the failing frame.
5. Released embeddings carry some signal for dark-edge prediction beyond name tokens on fastapi and rich (pending intervals).
6. The released graphs have no async function nodes; class nodes carry 40% of the calls edges in the example graph.

Not supported: H1's repo ordering. H3 as stated (embeddings do not beat the name baseline).
Not tested: H4 (agent with tracing tool vs static tools).

Do not claim: anything per-repo about requests or httpx; that dark edges are "dynamic dispatch" (mechanism untested); that oracle results show what an
agent can do; causal readings of the Spearman result; generalisation beyond three repos, one tracer configuration and tasks whose environment could be built (103 of 129).

## 9. Threats to validity (all to be stated)

- Selection: 26 tasks without a valid pre/post pair; 14 more without a gold function in the graph; task sets are not random.
- Graph artifacts: missing async nodes, class hubs, no decorator text, sparse `calls`-only edges; mapping by dotted-path suffix.
- Gold-function definition: old-side lines only; 41% of hunks unmapped; added lines anchored by indentation heuristic.
- Baseline design: issue-symbol seeding (exact id, dotted suffix, names matching <=3 nodes), edges of any type, undirected; BM25 own implementation.
- Tracing: only the tests that run in the pre/post pair; nested functions never map; lifted edges via framework frames.
- Statistics: no multiple-comparison correction; bootstrap with 5 tasks unreliable; pooled link-prediction AUCs without intervals (until extra 2).
- Exploratory choices made after seeing data: seeded-only subset, same-class and dunder exclusions, class-hub removal.

## 10. Open items before writing

1. Read HARNESS_README.md on how calls edges and class nodes are built; confirm or drop the self-call explanation.
2. Decide which dark-fraction definition is the headline (the protocol definition is the default; show the strict one next to it).
3. Run the three extras and fill sections 3, 6, 7.
4. If time allows: read decorators from the snapshots to make the cause labels meaningful (not done).
