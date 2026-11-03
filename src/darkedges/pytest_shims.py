"""pytest plugin loaded with `-p darkedges.pytest_shims` during tracing.
 1. registers ini options of plugins that are absent from the wheelhouse (pytest-timeout) so
    `--strict-config` / filterwarnings=error projects do not abort with "Unknown config option";
 2. per-test wall-clock timeout (SIGALRM, POSIX main thread): a test that hangs is reported as FAILED
    instead of killing the whole run (env DARKEDGES_TEST_TIMEOUT seconds, default 120, 0 = off)."""
import os
import signal

try:
    import pytest
except Exception:                      # imported only inside pytest; keep import errors loud elsewhere
    pytest = None

_INI = ("timeout", "timeout_method", "timeout_func_only")


def pytest_addoption(parser):
    for name in _INI:
        try:
            parser.addini(name, "darkedges shim for pytest-timeout option", default=None)
        except ValueError:             # already registered by the real plugin
            pass


def _limit():
    try:
        return float(os.environ.get("DARKEDGES_TEST_TIMEOUT", "120"))
    except ValueError:
        return 120.0


if pytest is not None:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(item):
        lim = _limit()
        armed = False
        if lim > 0 and hasattr(signal, "SIGALRM"):
            def _handler(signum, frame):
                raise TimeoutError(f"darkedges per-test timeout after {lim:g}s")
            try:
                old = signal.signal(signal.SIGALRM, _handler)
                signal.setitimer(signal.ITIMER_REAL, lim)
                armed = True
            except ValueError:         # not in main thread
                pass
        try:
            yield
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
