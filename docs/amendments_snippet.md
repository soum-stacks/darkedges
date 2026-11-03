## 2026-11-03 Amendment A1 (POST-HOC: made after the first full tracing run was seen)
Validity: in addition to the strict rule (>=1 failed pre; 0 failed and >=1 passed post), a task is counted valid_f2p when >=1 test
fails in the pre run and passes in the post run (SWE-bench style). Tests failing in both runs are environment-limited (missing fixtures
or packages) and are excluded from failing/passing sets. Reason: wheelhouse lacks packages (httpbin, trustme, ...) so unrelated tests fail
in both runs. Report BOTH counts in the paper.

## 2026-11-03 Amendment A2 (environment, post-hoc)
Stand-in packages `inline_snapshot` and `dirty_equals` (src/darkedges/shims, simplified comparison semantics, last on PYTHONPATH, used only
when absent) and plugin darkedges.pytest_shims (registers pytest-timeout ini options; per-test timeout 120 s, a hanging test counts as failed).
`--continue-on-collection-errors` added. Tasks run with shims are listed in data/traces/_env.jsonl (shims_on_path).
