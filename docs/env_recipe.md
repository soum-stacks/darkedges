# Per-task test environment recipe (verified 31 Oct 2026)

Where: Kaggle **CPU** notebook (Python 3.13.x), Internet off, inputs = main competition + private `darkedges-data`. Code: `notebooks/T0_8c_env_wheelhouse_first.ipynb`.

## Why not the obvious ways
- Competition `sandbox/setup.py` run in the Kaggle notebook: fastapi_14492 failed at collection (`Router.__init__() got an unexpected keyword argument 'on_startup'`) in both pre/post runs. Likely a newer starlette from the notebook's global environment shadowing the installed one (not confirmed). It worked for rich/httpx/requests.
- Strict `pip install --no-index --find-links wheels -r reqs` (even with `--target`): fails because the wheelhouse lacks transitive dependencies (e.g. typing-inspection, attrs). The real sandbox image supplies those from its base environment.

## Recipe (works)
1. Extract `snapshots/<id>.tgz` into a fresh dir (`tar -xzf`), set git identity, apply `test_patch` (pre) or `patch` then `test_patch` (post) with `git apply` (fallbacks `--3way`, `patch -p1`).
2. Requirements = project dependencies + optional-dependencies/dependency-groups named test|tests|testing|dev|all|standard + poetry deps (caret/tilde converted) + requirements*.txt; drop the project itself and `pytest` (the notebook's pytest is used).
3. Mini-resolver: index compatible wheels (`packaging.tags.sys_tags`), pick the highest wheel satisfying the accumulated specifiers, follow `Requires-Dist` from wheel METADATA (markers evaluated, extras ignored); dependencies with no wheel are skipped (the notebook's global env supplies them).
4. `pip install --no-deps --no-index --target <task>_deps <chosen wheels>`; copy `docker/imp.py` and `docker/telnetlib.py` shims into it.
5. Run `PYTHONPATH=<tracelens src>:<deps>:<repo>:<repo>/src python -m tracelens.run --root <repo> --out trace.json -- pytest <test files> -p no:anyio -p no:cacheprovider -q -k '<added test names>'` (and the same without the tracer for timing).
6. Valid task = pre run has at least one failed test and post run has none failed and at least one passed.

## Verified (Kaggle CPU, Python 3.13)
fastapi_14492, fastapi_9425, rich_3278, httpx_3672, requests_6644: all PASS. Different fastapi commits get different starlette (0.50.0 vs 0.48.0). Import origins confirmed per-task wheels load first. Tracing overhead about 0.9x to 1.3x on these short test runs. Setup 1 to 5 s per task with this recipe.

## Known risks
- Tasks whose tests need packages with no wheel and absent from the notebook environment (e.g. requests tests using httpbin/trustme) will fail collection: they count as invalid, report them.
- Only small test files were exercised. Large fastapi test files may be slower; keep the 900 s per-run timeout.
- Wheels are chosen greedily (no backtracking); check `envs.jsonl` notes for "no wheel satisfies".
