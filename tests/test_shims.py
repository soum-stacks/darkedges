import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(SRC / "darkedges" / "shims"))


def test_matchers():
    from dirty_equals import (AnyThing, IsDict, IsInt, IsList, IsOneOf, IsPartialDict, IsStr, IsUrl)
    from inline_snapshot import snapshot
    r = {"a": 1, "b": [1, 2], "c": "xyz"}
    assert r == snapshot({"a": 1, "b": [1, 2], "c": "xyz"})
    assert r == IsDict({"a": 1, "b": IsList(1, 2), "c": IsStr(r"x.*")})
    assert r == (IsDict({"a": 2}) | IsDict({"a": 1, "b": [1, 2], "c": AnyThing()}))
    assert r != IsDict({"a": 2, "b": 1, "c": 1}) and r == IsPartialDict(a=1)
    assert 5 == IsOneOf(1, 5) and 5 != IsOneOf(1, 2) and True != IsInt() and "http://x" == IsUrl()


def test_pytest_plugin_registers_ini_options():
    from darkedges import pytest_shims

    class P:
        def __init__(self): self.names = []
        def addini(self, name, *a, **k):
            if name in self.names: raise ValueError("dup")
            self.names.append(name)
    p = P(); pytest_shims.pytest_addoption(p); pytest_shims.pytest_addoption(p)   # 2nd call: swallowed
    assert p.names == ["timeout", "timeout_method", "timeout_func_only"]


if __name__ == "__main__":
    test_matchers(); test_pytest_plugin_registers_ini_options(); print("ok shims")
