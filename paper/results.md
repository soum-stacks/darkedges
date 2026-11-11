
# T2 notes (6 Nov 2026)

Working notes on the T2 analysis, not Writeup text. Numbers: results/hypotheses_summary.md (strict set), sens_f2p/results/hypotheses_summary.md,
results/callee_split.csv, results/linkpred_bootstrap.csv. Details of every choice: docs/t2_choices.md.

## 0. Setup

- Primary set "strict": pre run has >=1 failing test, post run (gold patch) has none and >=1 pass. 103 tasks (fastapi 58, rich 39, requests 5, httpx 1);
  89 have a gold function in the graph and are ranked. Sensitivity set "f2p": >=1 failing pre test, 120 tasks (60/47/12/1), 104 ranked.
- Gold functions: old-side lines of the gold patch -> enclosing function. 112 of 129 tasks have one; 288 of 695 hunks (41%) map to no graph node.
- Seed 0; 100 tie-break shuffles; task-clustered bootstrap (10,000; 1,000 for link prediction); exact McNemar. Python 3.12.16, numpy 2.5.4,
  scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1. Same-class / class-hub variants and the seeded-only subset were chosen after seeing data (exploratory).

## 1. Facts about the released graphs

- Example graph: 4,256 nodes, only `calls` edges (2,422), 78% of nodes have test-like ids. No `async def` node in any repo's graphs. Node `name` field = full id.
  Decorator lines almost never kept (18/3,593 function-like nodes), so `decorator`/`property` cause labels are dead; only dunder / lifted / other mean anything.
- 40% of `calls` edges in that graph (2,184/5,460) start at a class node: class nodes carry calls made inside their methods.
- HARNESS_README: graphs are "AST/dependency" graphs; the agent tool get_code_neighbors returns callers and callees for a node, including class nodes
  (its example is `fastapi.applications.FastAPI`, 18 neighbours). Nothing about class aggregation or async. So the agent sees the class hubs; keeping them in the static baselines is faithful.
- Call-weighted mapping coverage: rich/requests/httpx 1.000, fastapi 0.934 (unmapped: async `__call__` methods).

## 2. Dark edges are mostly calls into methods (main finding)

Dark edge: runtime edge between mapped repo functions (direct or lifted), no `calls` edge caller->callee. Unit: unique pair per task, task-macro mean.

| repo (tasks) | headline             | excl. same-class | excl. same-class and dunder callee |
| ------------ | -------------------- | ---------------- | ---------------------------------- |
| rich (39)    | 0.805 [0.752, 0.843] | 0.678            | 0.554 [0.516, 0.588]               |
| requests (5) | 0.517 [0.444, 0.591] | 0.411            | 0.287                              |
| fastapi (58) | 0.473 [0.440, 0.509] | 0.421            | 0.249 [0.226, 0.269]               |

Dark fraction by callee type (strict set, task-macro [CI]; share = share of the repo's pairs):

| callee                | fastapi                             | requests    | rich                       |
| --------------------- | ----------------------------------- | ----------- | -------------------------- |
| module-level function | 0.140 [0.125, 0.154] (66% of pairs) | 0.123 (52%) | 0.167 [0.135, 0.211] (12%) |
| method, same class    | 1.000 (8%)                          | 1.000 (22%) | 1.000 (44%)                |
| method, other class   | 0.999 (25%)                         | 1.000 (26%) | 0.874 [0.859, 0.890] (45%) |

- Calls to module-level functions are mostly present in the graph (83-88% not dark in all three repos). Calls to methods are almost never present
  (99.9-100% dark in fastapi and requests, 87-100% in rich). Repo differences follow the share of method callees among executed pairs:
  rich 88%, requests 48%, fastapi 34%; same order as the dark fractions.
- H1 (fastapi highest, requests lowest) fails under every definition on both task sets. Observed order is rich > requests > fastapi (strict, f2p; one
  strict variant swaps the last two, within noise). Rich minus fastapi: -0.33 / -0.26 / -0.32 (headline / excl. same-class / strict), CIs exclude 0.
  f2p (requests has 12 tasks): requests 0.615 vs fastapi 0.472, difference 0.144 [0.076, 0.206].
- Reading: the measure mostly tracks how method-heavy the executed code is, not how dynamic it is. Mechanism (method calls only appear at class
  level in these graphs) is inferred from examples and the 40% figure; the edge-kind table from scripts/10_method_share.py is the direct check (PENDING).
- Pooled: 14,059 pairs, 10,067 dark; same-class 46%, dunder callee 31% (overlap not computed). Class nodes as intermediates: 67% of dark edges are
  2 hops away; without them 5% (92% have no route within 3 hops). The near-miss rate is a class-node effect, do not use it as a metric.
- Lifted edges (through framework frames): 428 dark vs 3 static. Dark by construction; only fastapi and requests have them (rich headline = direct-only).
- Caveats: edges with async endpoints are invisible (fastapi most affected, bias direction unknown); requests = 5 tasks (strict); rich holds 70% of pairs, so pooled shares describe rich.

## 3. H2: localisation (recall@20, task-macro mean)

| method              | strict, 89 tasks | f2p, 104 tasks |
| ------------------- | ---------------- | -------------- |
| ochiai (oracle)     | 0.631            | 0.586          |
| stack_dist (oracle) | 0.573            | 0.529          |
| bm25                | 0.461            | 0.481          |
| static_k2           | 0.260            | 0.269          |

- Trace methods beat static 2-hop by far: ochiai +0.371 [0.256, 0.485] (strict) / +0.317 [0.200, 0.428] (f2p); stack_dist +0.313 [0.183, 0.441] /
  +0.260 [0.141, 0.378]; McNemar p<0.001 in all four. H2 (>=10 points) holds on both sets.
- Ochiai over BM25: +0.170 [0.031, 0.306] (p=0.011) strict but +0.105 [-0.031, 0.237] (p=0.105) f2p: not stable, do not claim. stack_dist vs BM25 n.s. on both.
  BM25 beats static_k2 by about 0.20 on both. At K=5 the three non-static methods overlap (0.32-0.42); BM25 leads on the 63 seeded strict tasks (0.413 vs ochiai 0.361, stack_dist 0.371).
- Ochiai and stack_dist use the failing tests that define the task: ceilings, not what an agent has. Static baseline: 26/89 (strict) and 31/104 (f2p) ranked
  tasks have no issue-text seed and count as misses; on the 63 seeded strict tasks at K=20: ochiai 0.628, bm25 0.540, stack_dist 0.522, static_k2 0.367 (post hoc subset).
- 20 tasks have no passing test (Ochiai ties). Only the two static_k2 contrasts at K=20 were pre-specified; no multiple-comparison correction.

## 4. Gold edit vs traces

| fraction of tasks                              | strict (89) | f2p (104) |
| ---------------------------------------------- | ----------- | --------- |
| gold executed by failing tests                 | 0.831       | 0.808     |
| ...and by no passing test                      | 0.292       | 0.250     |
| ...reachable only via dark edges               | 0.202       | 0.212     |
| gold within 3 static hops of the failing frame | 0.236       | 0.260     |

- The edit location is run by the failing tests in about 8 of 10 tasks, but lies in a static 3-hop neighbourhood of the failing frame (class hubs included, test frames included) in about 2.5 of 10.
- dark_only by repo: rich 0.33 (f2p 0.37), fastapi 0.14, requests 0. Untested idea: gold functions in rich are mostly methods, which the graph does not link.
- Spearman dark fraction vs dark_only: 0.25 (p=0.017) strict, 0.30 (p=0.002) f2p. Both come from the same edge labels and rich confounds; not independent evidence.

## 5. H3: dark-edge prediction (AUC, leave-one-repo-out, task-clustered 95% CI, strict set)

| held-out           | variant       | emb_cos   | name_jaccard | gap emb - name                       | ppr       | logreg    | logreg w/o emb |
| ------------------ | ------------- | --------- | ------------ | ------------------------------------ | --------- | --------- | -------------- |
| fastapi            | default       | 0.703     | 0.580        | +0.122 [0.112, 0.134]                | 0.765     | 0.782     | 0.790          |
| fastapi            | no class hubs | 0.711     | 0.579        | +0.132 [0.122, 0.142]                | 0.566     | 0.751     | 0.695          |
| fastapi            | cross-class   | 0.697     | 0.601        | +0.096 [0.084, 0.109]                | 0.551     | 0.735     | 0.666          |
| rich               | default       | 0.695     | 0.554        | +0.141 [0.132, 0.152]                | 0.764     | 0.778     | 0.792          |
| rich               | no class hubs | 0.700     | 0.554        | +0.146 [0.136, 0.156]                | 0.521     | 0.759     | 0.719          |
| rich               | cross-class   | 0.629     | 0.554        | +0.076 [0.067, 0.085]                | 0.521     | 0.681     | 0.609          |
| requests (5 tasks) | all three     | 0.61-0.73 | 0.62-0.73    | +0.008 / -0.020 / -0.008, CIs span 0 | 0.46-0.75 | 0.71-0.85 | 0.70-0.86      |

- H3 (embeddings do not beat a name-token baseline) is not supported on fastapi and rich: embeddings win in all three variants, CIs exclude 0, on both task sets.
  Requests is inconclusive (strict n=5 CIs span 0; f2p n=12: +0.033, +0.043, -0.023, point estimates only).
- Structural scorers (common neighbours, Adamic-Adar, PPR) reach 0.69-0.77 only through class hubs: without class nodes they fall to 0.51-0.57
  (same pattern on f2p). Embeddings then beat PPR by 0.11-0.18 AUC on fastapi and rich (CIs exclude 0).
- Embeddings add to the other scorers only after the hubs are removed: logreg minus logreg-without-embeddings +0.04 to +0.07 on fastapi and rich
  (no hubs, cross-class), but about -0.01 with hubs present.
- Effect sizes are modest: best cross-class AUC 0.68-0.74. Hits@10 is mixed (name tokens beat embeddings on requests, and marginally on rich cross-class).
- CIs resample tasks, but tasks share code and the negative sample is fixed per task, so they are optimistic.

## 6. Claims

Supported (with the conditions above):

1. Most dark edges are calls into methods, which the graphs almost never link; calls to module-level functions are mostly linked.
2. Rich's dark fraction is far above fastapi's (all definitions, both sets); the H1 repo ordering is not supported. Nothing per-repo for requests/httpx.
3. Failing-test-based ranking beats a static issue-symbol neighbourhood at K=20 by 26-37 points (oracle ceilings); the Ochiai-BM25 gap is not stable.
4. The gold function is executed by the failing tests in about 81-83% of tasks and sits within 3 static hops of the failing frame in 24-26%.
5. Released embeddings predict dark edges beyond name tokens on fastapi and rich; structural scores only look good because of class hubs.
6. Released graphs: no async nodes, class nodes carry about 40% of calls edges (one graph), decorators mostly absent.
   Not supported: H1 ordering; H3 as stated. Not tested: H4 (agent with tracing).
   Do not claim: dynamic-dispatch explanations, agent-usable gains from oracle numbers, causal readings of Spearman, results beyond three repos / 103 buildable tasks.

## 7. Threats

- Selection: 26 tasks without a valid run pair, 14-16 more without a gold function in the graph; gold edits outside functions are dropped.
- Graph artifacts: async, class hubs, no decorator text, `calls`-only edges in the graph checked, dotted-suffix mapping.
- Baseline design: issue-symbol seeding (names matching <=3 nodes), any edge type, undirected; own BM25. Tie-averaged hits, hit_bin >= 0.5 for McNemar.
- Tracing: only the tests in the run pair; nested functions never map; one tracer configuration.

## 8. Open

1. Run scripts/10_method_share.py: edge-kind table (is the calls-into-method share really ~0?), per-task method share vs dark fraction, figure F5.
2. Pick the headline dark-fraction definition (protocol definition by default; show the by-callee-type view next to it).
3. Optional: read decorators from the snapshots so causes mean something. H4 (Phase 3) not started.
