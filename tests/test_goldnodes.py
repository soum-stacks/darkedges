import importlib.util
import io
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gold03b", ROOT / "scripts" / "03b_gold_and_embeddings.py")
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)
sys.path.insert(0, str(ROOT / "src"))
from darkedges.compare import map_node  # noqa: E402

SRC = """import os

def helper(x):
    y = x + 1
    return y

class Widget:
    attr = 1

    def render(self):
        def inner():
            return 2
        return inner()

    def other(self):
        return 3
"""
LINES = SRC.splitlines()
NODES = {"pkg.core.helper", "pkg.core.Widget", "pkg.core.Widget.render", "pkg.core.Widget.other"}


def ctx(n):
    return " " + LINES[n - 1]


def mk(old_start, old_len, new_len, body, path="pkg/core.py"):
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -{old_start},{old_len} +{old_start},{new_len} @@\n" + "\n".join(body) + "\n")


def run(patch):
    return g3.gold_for_patch(patch, lambda p: SRC if p == "pkg/core.py" else None, NODES)


def test_modified_line_in_function():
    r = run(mk(3, 3, 3, [ctx(3), "-    y = x + 1", "+    y = x + 2", ctx(5)]))
    assert r["gold_node_ids"] == ["pkg.core.helper"] and r["n_hunks"] == 1 and r["unmapped_hunks"] == 0


def test_nested_function_falls_back_to_enclosing():
    r = run(mk(11, 3, 3, [ctx(11), "-            return 2", "+            return 5", ctx(13)]))
    assert r["gold_node_ids"] == ["pkg.core.Widget.render"]


def test_pure_addition_inside_method():
    r = run(mk(15, 2, 3, [ctx(15), ctx(16), "+        return 4"]))
    assert r["gold_node_ids"] == ["pkg.core.Widget.other"]


def test_pure_addition_between_functions_is_unmapped():
    r = run(mk(5, 2, 5, [ctx(5), "+", "+def newfn():", "+    pass", ctx(6)]))
    assert r["gold_node_ids"] == [] and r["unmapped_hunks"] == 1


def test_class_level_change_is_unmapped():
    r = run(mk(7, 3, 3, [ctx(7), "-    attr = 1", "+    attr = 2", ctx(9)]))
    assert r["gold_node_ids"] == [] and r["unmapped_hunks"] == 1


def test_multi_file_and_nonpy_and_new_file():
    p = mk(3, 3, 3, [ctx(3), "-    y = x + 1", "+    y = x + 2", ctx(5)])
    p += "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1,1 +1,1 @@\n-a\n+b\n"
    p += "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+x = 1\n"
    r = run(p)
    assert r["gold_node_ids"] == ["pkg.core.helper"] and r["n_nonpy_files"] == 1 and r["n_hunks"] == 1


def test_tar_reader_and_mapping_agree(tmp_path):
    tgz = tmp_path / "s.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        data = SRC.encode()
        ti = tarfile.TarInfo("./pkg/core.py")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    read, tar = g3.make_tar_reader(tgz)
    assert read("pkg/core.py") == SRC and read("nope.py") is None
    tar.close()
    for rel, q in [("src/requests/api.py", "get"), ("fastapi/__init__.py", "X"), ("pkg/mod.py", "K.m")]:
        ns = {"requests.api.get", "fastapi.X", "pkg.mod.K.m"}
        assert g3.map_id(rel, q, ns) == map_node(rel, q, ns)
