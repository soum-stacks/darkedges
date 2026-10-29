"""pytest plugin: marks test boundaries for the tracer and records outcomes and repo failure frames."""
import os

import pytest


class TraceLensPlugin:
    def __init__(self, tracer):
        self.t = tracer

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self.t.begin_test(item.nodeid)
        yield
        self.t.end_test()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        rep = outcome.get_result()
        frames = []
        if call.excinfo is not None:
            for e in call.excinfo.traceback:
                try:
                    p = os.path.realpath(str(e.path))
                except Exception:
                    continue
                if p.startswith(self.t.root + os.sep):
                    frames.append([os.path.relpath(p, self.t.root).replace(os.sep, "/"), e.lineno + 1, e.name])
        if rep.when == "call" or (rep.when == "setup" and rep.outcome != "passed"):
            self.t.set_outcome(item.nodeid, rep.outcome, frames[-8:])
