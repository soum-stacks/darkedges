"""Plumbing test for darkedges.tracing with a STUB tracelens.run (real tracer/pytest not exercised here)."""
import json, subprocess, sys, tarfile, tempfile, textwrap, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darkedges import tracing, envbuild  # noqa: E402

STUB = '''
import json, sys, pathlib
a = sys.argv[1:]
root = pathlib.Path(a[a.index("--root")+1]); out = a[a.index("--out")+1]
fixed = "return a + 2" in (root / "pkg" / "mod.py").read_text()
tests = {"tests/test_mod.py::test_a": {"outcome": "passed" if fixed else "failed", "nodes": [0], "edges": []},
         "tests/test_mod.py::test_b": {"outcome": "passed", "nodes": [0], "edges": []}}
json.dump({"root": str(root), "tests": tests, "nodes": [], "edges": []}, open(out, "w"))
'''

BASE = "def top(a):\n    return a + 1\n"
FIXED = "def top(a):\n    return a + 2\n"
TEST = "from pkg.mod import top\n\ndef test_a():\n    assert top(1) == 3\n\ndef test_b():\n    assert top(0) == 1\n"


def _git_diff(old: dict, new: dict, tmp: Path) -> str:
    import uuid; d = tmp / ("g" + uuid.uuid4().hex[:6]); d.mkdir()
    for sub in ("a", "b"):
        for k, v in (old if sub == "a" else new).items():
            p = d / sub / k; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(v)
    (d / "a").mkdir(exist_ok=True); (d / "b").mkdir(exist_ok=True)
    r = subprocess.run(["git", "diff", "--no-index", "--no-color", "a", "b"], cwd=d, capture_output=True, text=True)
    return r.stdout.replace("a/a/", "a/").replace("b/b/", "b/")


def test_end_to_end():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        comp = t / "comp"
        for s in ("snapshots", "graphs", "wheels", "docker"):
            (comp / s).mkdir(parents=True)
        (comp / "docker" / "imp.py").write_text("#"); (comp / "docker" / "telnetlib.py").write_text("#")
        snap_src = t / "snap"; (snap_src / "pkg").mkdir(parents=True)
        (snap_src / "pkg" / "mod.py").write_text(BASE); (snap_src / "pyproject.toml").write_text('[project]\nname="pkg"\ndependencies=[]\n')
        with tarfile.open(comp / "snapshots" / "demo_1.tgz", "w:gz") as tf:
            for p in snap_src.rglob("*"):
                tf.add(p, arcname=str(p.relative_to(snap_src)))
        gold = _git_diff({"pkg/mod.py": BASE}, {"pkg/mod.py": FIXED}, t)
        tpatch = _git_diff({}, {"tests/test_mod.py": TEST}, t)
        task = {"instance_id": "demo_1", "repo": "x/pkg", "base_commit": "c" * 40, "patch": gold, "test_patch": tpatch,
                "problem_statement": "", "hints_text": "", "created_at": ""}
        (comp / "tasks.jsonl").write_text(json.dumps(task) + "\n")
        (comp / "graphs" / ("pkg_" + "c" * 40 + ".json")).write_text(json.dumps(
            {"directed": True, "nodes": [{"id": "pkg.mod.top", "name": "top", "text": "x" * 60}], "edges": []}) + " " * 100)
        stubdir = t / "stub"; (stubdir / "tracelens").mkdir(parents=True)
        (stubdir / "tracelens" / "__init__.py").write_text(""); (stubdir / "tracelens" / "run.py").write_text(STUB)
        tracing.SRC_DIR = stubdir
        loc = tracing.locate(comp)
        out = t / "out"
        tracing.run_all(loc, out, t / "work", log=lambda *a: None)
        st = [json.loads(l) for l in (out / "traces" / "_status.jsonl").read_text().splitlines()]
        assert [(r["phase"], r["ok"], r["tests_failed"], r["tests_passed"]) for r in st] == \
            [("pre", True, 1, 1), ("post", True, 0, 2)], st
        gn = json.loads((out / "results" / "gold_nodes.jsonl").read_text())
        assert gn["gold_node_ids"] == ["pkg.mod.top"], gn
        rep = tracing.validity_report(out, loc["tasks"], gate=1, log=lambda *a: None)
        assert rep["valid"]["valid"] == 1 and rep["valid_f2p"]["valid"] == 1
        v = json.loads((out / "traces" / "_valid.jsonl").read_text())
        assert v["f2p"] == ["tests/test_mod.py::test_a"], v
        n_before = len((out / "traces" / "_status.jsonl").read_text().splitlines())
        tracing.run_all(loc, out, t / "work", log=lambda *a: None)             # resume: nothing re-run
        assert len((out / "traces" / "_status.jsonl").read_text().splitlines()) == n_before
        assert not (t / "work" / "demo_1").exists()


if __name__ == "__main__":
    test_end_to_end(); print("ok test_end_to_end")
