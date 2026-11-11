---
name: tracelens
description: Run a reproduction script or tests under a runtime call tracer and get a ranked list of the repository functions that actually executed (failure stack first), including method calls the code graph lacks. Use it after you have a failing reproduction and before editing.
---

# TraceLens: find where to edit by tracing what really runs

The precomputed code graph (get_code_neighbors) mostly lacks calls into methods. TraceLens records the
calls that really happen when you run a reproduction, so it shows the path from the entry point to the
failure, including self.method() calls and calls made through framework code.

## Steps
1. Write a short reproduction of the issue with write_file at `tests/_repro.py` (files under tests/ are
   removed from your patch automatically). It must import the package from the repository and fail
   (an assertion or exception) while the bug is present.
2. Run the tracer with run_skill_script or run_command:
   `python3 scripts/trace_run.py tests/_repro.py`
   To trace existing tests instead: `python3 scripts/trace_run.py -m pytest tests/test_x.py -q -x`
3. Read the top SUSPECTS with read_file (file:line is given). Functions on the failure stack come first.
4. Fix the library code, re-run the reproduction to confirm, then call submit_patch.

## Notes
- Offline, standard library only; the traced command is limited to 85 seconds.
- A timeout often means the bug is a hang: trace a smaller input.
- Output is capped at about 4,000 characters.
