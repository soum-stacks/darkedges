import asyncio
import importlib
import json
import os
import subprocess
import sys
import textwrap

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))
from tracelens import Tracer  # noqa: E402
from tracelens.mapping import to_node_id, module_candidates  # noqa: E402


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    ext = tmp_path / "ext"          # NOT under the repo root: acts as a framework
    ext.mkdir()
    (ext / "framework.py").write_text("def dispatch(f):\n    return f()\n")
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "core.py").write_text(textwrap.dedent('''
        import asyncio, framework

        def b():
            return 1

        def a():
            return b()

        def deco(f):
            def w(*x):
                return f(*x)
            return w

        @deco
        def d():
            return b()

        class K:
            def m(self):
                return a()

        def cb():
            return b()

        def via_framework():
            return framework.dispatch(cb)

        def gen():
            yield b()

        def comp():
            return [b() for _ in range(2)]

        async def ac():
            return b()

        async def amain():
            return await ac()

        def entry():
            a(); d(); K().m(); via_framework(); list(gen()); comp()
            asyncio.run(amain())
    '''))
    return repo, ext


def run_traced(tmp_path, backend_monitoring=True):
    repo, ext = make_repo(tmp_path)
    sys.path[:0] = [str(repo), str(ext)]
    for m in ("pkg", "pkg.core", "framework"):
        sys.modules.pop(m, None)
    try:
        core = importlib.import_module("pkg.core")
        with Tracer(str(repo), use_monitoring=backend_monitoring) as tr:
            core.entry()
    finally:
        sys.path[:2] = []
    return tr.to_dict()


def edge_set(d):
    nodes = {n["i"]: n["qualname"] for n in d["nodes"]}
    return {(nodes.get(e["caller"]), nodes[e["callee"]], e["kind"]) for e in d["edges"]}


@pytest.mark.parametrize("mon", [True, False])
def test_edges(tmp_path, mon):
    if mon and not hasattr(sys, "monitoring"):
        pytest.skip("needs Python 3.12+")
    d = run_traced(tmp_path, mon)
    E = edge_set(d)
    assert ("a", "b", "direct") in E
    assert ("K.m", "a", "direct") in E
    assert ("entry", "a", "direct") in E and ("entry", "K.m", "direct") in E
    # decorator wrapper: d is replaced by w, so entry -> deco.<locals>.w -> d -> b
    assert ("deco.<locals>.w", "d", "direct") in E
    # framework in between: via_framework -> (ext framework.dispatch) -> cb  == lifted
    assert ("via_framework", "cb", "lifted") in E
    assert ("cb", "b", "direct") in E
    # generator and list comprehension attribute to enclosing function
    assert ("gen", "b", "direct") in E
    assert ("comp", "b", "direct") in E
    assert not any("<listcomp>" in (c or "") or "<listcomp>" in e for c, e, _ in E)
    # async
    assert ("amain", "ac", "direct") in E and ("ac", "b", "direct") in E
    # stack discipline: no spurious edges from b (leaf)
    assert not any(c == "b" for c, _, _ in E)


def test_external_code_not_recorded(tmp_path):
    d = run_traced(tmp_path)
    assert all("framework" not in n["relpath"] for n in d["nodes"])


def test_mapping():
    ids = {"fastapi.routing.APIRouter.get", "fastapi.routing", "pkg.core.a"}
    assert to_node_id("src/fastapi/routing.py", "APIRouter.get", ids) == "fastapi.routing.APIRouter.get"
    assert to_node_id("pkg/core.py", "a", ids) == "pkg.core.a"
    assert to_node_id("fastapi/routing.py", "<module>", ids) == "fastapi.routing"
    assert to_node_id("pkg/core.py", "deco.<locals>.w", ids) is None
    assert module_candidates("a/b/__init__.py") == ["a.b", "b"]


def test_cli_script(tmp_path):
    repo, ext = make_repo(tmp_path)
    script = tmp_path / "repro.py"       # outside root on purpose, like /tmp/repro.py
    script.write_text("import sys; sys.path[:0]=[%r,%r]\nimport pkg.core as c\nc.a()\nraise RuntimeError('boom')\n"
                      % (str(repo), str(ext)))
    out = tmp_path / "t.json"
    env = dict(os.environ, PYTHONPATH=os.path.abspath(SRC))
    r = subprocess.run([sys.executable, "-m", "tracelens.run", "--root", str(repo), "--out", str(out),
                        "--", str(script)], env=env, capture_output=True, text=True)
    assert r.returncode == 1, r.stderr
    d = json.load(open(out))
    assert d["exit_code"] == 1
    assert any(n["qualname"] == "a" for n in d["nodes"])
    from tracelens.summarize import summarize
    s = summarize(str(out))
    assert "FIRST-CALLED" in s and len(s) <= 4500
