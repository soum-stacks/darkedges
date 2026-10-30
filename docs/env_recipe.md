
# Per-task environment recipe

How to run one task's tests, before and after the gold patch, under the tracer, in a Kaggle notebook.
Verified only for `requests_6644` (2026-10-30). Everything marked **untested** has not been run.

## Where things are (Kaggle notebooks with the competition attached)

- Competition folder: `/kaggle/input/competitions/gemma-4-developer-agent/`
  - `tasks.jsonl`, `snapshots/<instance_id>.tgz`, `graphs/`, `embeddings/`, `sample_submission/`
  - `docker/Dockerfile.sandbox`, `sandbox/setup.py`, `wheels/`
- This repo as a private Kaggle dataset, copied into the working folder (the input folder is read-only):
  ```
  !rm -rf /kaggle/working/darkedges && mkdir -p /kaggle/working/darkedges
  !cp -r /kaggle/input/datasets/<owner>/<dataset>/darkedges/* /kaggle/working/darkedges/
  ```

  Re-copy after each new dataset version. Keep the dataset **Private**: it contains `data/slim/` graphs.
- Competition internet is off. CPU notebook: Python 3.13.15, pytest 8.4.2. GPU starter notebook: Python 3.12.x.

## Recipe (verified on requests_6644)

1. Extract `snapshots/<id>.tgz` into a fresh directory. If the archive has a single top-level folder, that folder is the repo root.
2. Phase **pre**: `git apply --whitespace=nowarn` the task's `test_patch`. Phase **post**: apply `patch` (gold), then `test_patch`. Always start each phase from a fresh extraction.
3. `PYTHONPATH=<root>/src:<root>:/kaggle/working/darkedges/src`. This makes the snapshot's own source win over any installed copy of the package. Pre failing and post passing proves it did.
4. Tests: take the files from the `+++ b/` lines of `test_patch`, then narrow with `-k` to the added or changed test names (regex on added `def test...` lines and on hunk headers). Flags: `-p no:cacheprovider -q -p no:anyio`.
5. Untraced timing: `python -m pytest <args>`.
6. Traced: `python -m tracelens.run --root <root> --out /tmp/<id>.<phase>.json -- pytest <args>`.
7. Acceptance: at least one new test is `failed` in the pre trace and `passed` in the post trace.

All of this is scripted in `scripts/02_env_spike.py`:

```
!python /kaggle/working/darkedges/scripts/02_env_spike.py requests_6644
```

Flags: `--keep-anyio` (drops `-p no:anyio`), `--whole-files` (runs the whole test files instead of `-k`, for a more realistic overhead), `--log PATH`.
The script was checked with a stub pytest and stub tracer on a toy repo, and the notebook cell it came from passed on Kaggle. The script itself has **not** been run on Kaggle yet.

## Result so far

| task          | pre      | post     | overhead    | notes                                            |
| ------------- | -------- | -------- | ----------- | ------------------------------------------------ |
| requests_6644 | failed   | passed   | 1.2x / 1.1x | 1.6 s untraced; no installs needed               |
| fastapi_14492 | untested | untested |             | try without `-p no:anyio` if async tests error |
| rich_3278     | untested | untested |             |                                                  |
| httpx_3672    | untested | untested |             | only httpx task in the set                       |

Record each result here, plus any module that was missing.

## Choosing spike tasks

`requests` tests that use the `httpbin` fixture need extra packages that are not in `/wheels`. Pick tasks whose `test_patch` does not mention `httpbin`. Without `httpbin`, the requests tasks are `requests_6644`, `7315`, `6629`, `7427`, `7309`.

## Sandbox facts (competition runtime)

- Image: `python:3.13-slim` plus git, patch, protobuf-compiler, pigz. Pip packages: pytest, pytest-timeout==2.1.0, typer, pdm-backend, setuptools, wheel, poetry-core, hatchling, flit-core, editables. `imp.py` and `telnetlib.py` shims are copied into site-packages. `TEST_TMPDIR=/tmp`, `WORKDIR /workspace`.
- `sandbox/setup.py` runs before tests: it reads pyproject, setup.cfg and requirements files, resolves wheels from `/wheels`, and installs the workspace package editable. Only its first 100 lines have been read. Read the rest if a repo's dependencies fail to resolve.
- `/wheels`: 124 files, 8 cp312, 8 cp313, the rest pure Python. pytest 6.2.5, 8.3.4, 9.1.1; anyio; sqlmodel. No pytest-httpbin, trustme, inline-snapshot or dirty-equals found by name.
- The harness creates a task venv without pip (`ensurepip` fails), so anything the agent runs must be stdlib only.
- Sample `eval_config.yaml`: 60 s command timeout, 10 tool calls, 1 minute, 50 turns. Under tracing, a test run must fit inside the command timeout.

## Known gaps (untested)

- Whether the Kaggle image has the dependencies for fastapi, rich and httpx at the task commits.
- Overhead on heavy test files (use `--whole-files`).
- Running the competition's `setup.py` itself to build a faithful environment.
