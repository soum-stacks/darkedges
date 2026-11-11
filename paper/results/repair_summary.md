# Static graph repair (exploratory, post hoc) - facts only

Task set strict; 103 tasks. Unit: unique runtime (caller, callee) pair per task; task-macro mean with 95% task-clustered CI.

## Dark fraction before / after repair
| Repo | tasks | original | R1 self-calls | R2 +attributes | R3 +constructors | share of dark pairs recovered (R1/R2/R3) |
|---|---|---|---|---|---|---|
| fastapi | 58 | 0.473 [0.440, 0.509] | 0.421 [0.390, 0.457] | 0.385 [0.352, 0.424] | 0.341 [0.321, 0.362] | 0.111/0.192/0.236 |
| httpx | 1 | 1.000 [1.000, 1.000] | 0.789 [0.789, 0.789] | 0.789 [0.789, 0.789] | 0.737 [0.737, 0.737] | 0.211/0.211/0.263 |
| requests | 5 | 0.517 [0.444, 0.591] | 0.366 [0.281, 0.451] | 0.366 [0.281, 0.451] | 0.324 [0.198, 0.451] | 0.379/0.379/0.483 |
| rich | 39 | 0.805 [0.752, 0.843] | 0.682 [0.640, 0.715] | 0.524 [0.491, 0.554] | 0.446 [0.417, 0.476] | 0.161/0.354/0.452 |

## Pooled by callee type (pairs, dark, recovered R1/R2/R3)
- function: pairs 3942, dark 587, recovered 0/0/0
- method_other_class: pairs 5462, dark 4825, recovered 4/4/736
- method_same_class: pairs 4655, dark 4655, recovered 1542/3268/3424

## Precision proxy (added edges whose two endpoints both ran; share observed as runtime pairs)
- R1: 1546/1774 = 0.871
- R2: 3272/3737 = 0.876
- R3: 4160/4822 = 0.863

## Static k-hop localisation recall@K (issue-symbol seeds), original vs repaired graph
| method@K | tasks | original | R3 | R3 - original [CI] | McNemar R3-only/orig-only, p | median candidates orig/R3 |
|---|---|---|---|---|---|---|
| static_k1@5 | 89 | 0.177 [0.106, 0.256] | 0.189 [0.115, 0.270] | +0.012 [-0.003, +0.039] | 1/1, p=1 | 8.0/9.0 |
| static_k1@10 | 89 | 0.210 [0.131, 0.298] | 0.231 [0.147, 0.320] | +0.021 [-0.001, +0.054] | 2/0, p=0.5 | 8.0/9.0 |
| static_k1@20 | 89 | 0.241 [0.157, 0.331] | 0.261 [0.172, 0.355] | +0.020 [-0.003, +0.055] | 2/0, p=0.5 | 8.0/9.0 |
| static_k1@50 | 89 | 0.265 [0.175, 0.360] | 0.286 [0.191, 0.382] | +0.021 [-0.003, +0.056] | 2/1, p=1 | 8.0/9.0 |
| static_k2@5 | 89 | 0.177 [0.106, 0.257] | 0.192 [0.118, 0.274] | +0.015 [+0.000, +0.041] | 2/0, p=0.5 | 47.0/51.0 |
| static_k2@10 | 89 | 0.215 [0.137, 0.302] | 0.232 [0.148, 0.321] | +0.016 [-0.003, +0.045] | 2/0, p=0.5 | 47.0/51.0 |
| static_k2@20 | 89 | 0.259 [0.175, 0.349] | 0.267 [0.180, 0.360] | +0.007 [-0.011, +0.032] | 2/0, p=0.5 | 47.0/51.0 |
| static_k2@50 | 89 | 0.328 [0.235, 0.425] | 0.337 [0.242, 0.435] | +0.008 [-0.010, +0.037] | 1/1, p=1 | 47.0/51.0 |
| static_k3@5 | 89 | 0.176 [0.105, 0.254] | 0.187 [0.115, 0.268] | +0.012 [-0.005, +0.038] | 2/1, p=1 | 86.0/109.0 |
| static_k3@10 | 89 | 0.215 [0.136, 0.303] | 0.231 [0.148, 0.321] | +0.017 [-0.003, +0.045] | 2/0, p=0.5 | 86.0/109.0 |
| static_k3@20 | 89 | 0.259 [0.175, 0.349] | 0.270 [0.183, 0.363] | +0.011 [-0.007, +0.036] | 2/0, p=0.5 | 86.0/109.0 |
| static_k3@50 | 89 | 0.338 [0.247, 0.434] | 0.346 [0.252, 0.444] | +0.008 [-0.006, +0.025] | 1/1, p=1 | 86.0/109.0 |

Added edges per graph: see repair_summary.json['graphs'].
