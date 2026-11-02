import subprocess, sys, tempfile, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darkedges import hunks  # noqa: E402

BASE = textwrap.dedent('''\
    import os

    X = 1

    def top(a):
        return a + 1

    class K:
        @property
        def p(self):
            return 1

        def m(self, v):
            def inner(z):
                return z * 2
            return inner(v)

        async def am(self):
            return 3

    def last():
        pass
''')
IDS = {"pkg.mod.top", "pkg.mod.K.p", "pkg.mod.K.m", "pkg.mod.K.am", "pkg.mod.last"}


def _repo(t):
    r = Path(t) / "repo"; (r / "src/pkg").mkdir(parents=True)
    (r / "src/pkg/mod.py").write_text(BASE)
    return r


def _diff(r, new_text, U=1):
    f = r / "src/pkg/mod.py"; old = f.read_text()
    if not (r / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=r, check=True)
        subprocess.run(["git", "add", "-A"], cwd=r, check=True)
        subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=n", "commit", "-qm", "x"], cwd=r, check=True)
    f.write_text(new_text)
    d = subprocess.run(["git", "diff", f"-U{U}"], cwd=r, capture_output=True, text=True, check=True).stdout
    f.write_text(old)                      # gold_nodes reads the BASE file
    return d


def test_modify_and_nested_and_decorator():
    with tempfile.TemporaryDirectory() as t:
        r = _repo(t)
        new = BASE.replace("return a + 1", "return a + 2").replace("return z * 2", "return z * 3") \
                  .replace("    @property\n", "    @property  # changed\n")
        res = hunks.gold_nodes_detailed(_diff(r, new), r, IDS)
        assert res["gold_node_ids"] == ["pkg.mod.K.m", "pkg.mod.K.p", "pkg.mod.top"], res
        assert res["unmapped_hunks"] == 0 and res["n_hunks"] == 2


def test_pure_insertion_inside_function_and_module_level():
    with tempfile.TemporaryDirectory() as t:
        r = _repo(t)
        new = BASE.replace("    return a + 1\n", "    b = a\n    return a + 1\n").replace("X = 1\n", "X = 1\nY = 2\n")
        for U in (0, 3):
            res = hunks.gold_nodes_detailed(_diff(r, new, U), r, IDS)
            assert res["gold_node_ids"] == ["pkg.mod.top"], res
            assert res["unmapped_hunks"] == (1 if U == 0 else 0), res   # U3 merges both edits into one hunk


def test_new_function_after_last_is_unmapped_and_missing_ids_reported():
    with tempfile.TemporaryDirectory() as t:
        r = _repo(t)
        res = hunks.gold_nodes_detailed(_diff(r, BASE + "\ndef brand_new():\n    pass\n"), r, IDS)
        assert res["gold_node_ids"] == [] and res["unmapped_hunks"] == 1
        res = hunks.gold_nodes_detailed(_diff(r, BASE.replace("return 3", "return 4")), r, {"other.id"})
        assert res["gold_node_ids"] == [] and res["unmatched"] == ["src/pkg/mod.py::K.am"]


def test_set_wrapper_and_non_py_ignored():
    with tempfile.TemporaryDirectory() as t:
        r = _repo(t)
        d = _diff(r, BASE.replace("pass", "return None"))
        d += "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n"
        assert hunks.gold_nodes(d, r, IDS) == {"pkg.mod.last"}


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
