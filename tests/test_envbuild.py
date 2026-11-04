"""Mock-wheelhouse tests for darkedges.envbuild (no network, no real wheels)."""
import sys, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darkedges import envbuild as eb  # noqa: E402


def fake_wheel(d: Path, name: str, ver: str, requires=(), tag="py3-none-any"):
    fn = d / f"{name}-{ver}-{tag}.whl"
    di = f"{name}-{ver}.dist-info"
    meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n" + "".join(
        f"Requires-Dist: {r}\n" for r in requires)
    with zipfile.ZipFile(fn, "w") as z:
        z.writestr(f"{name}/__init__.py", f"VERSION = '{ver}'\n")
        z.writestr(f"{di}/METADATA", meta)
        z.writestr(f"{di}/WHEEL", "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(f"{di}/RECORD", "")
    return fn


def make_house(tmp: Path) -> eb.Wheelhouse:
    wd = tmp / "wheels"; wd.mkdir()
    fake_wheel(wd, "pytest", "8.0", ["iniconfig", "pluggy>=1"])
    fake_wheel(wd, "iniconfig", "2.0")
    fake_wheel(wd, "pluggy", "1.5")
    fake_wheel(wd, "starlette", "0.37", ["anyio>=3.4,<5"])
    fake_wheel(wd, "starlette", "0.50", ["anyio>=4"])
    fake_wheel(wd, "anyio", "3.7")
    fake_wheel(wd, "anyio", "4.4", ["idna>=2.8", "typing-inspection ; python_version < '3.0'"])
    fake_wheel(wd, "idna", "3.7")
    fake_wheel(wd, "pydantic", "2.0", ["annotated-types>=0.4", "extra-only ; extra == 'email'"])
    fake_wheel(wd, "annotated_types", "0.7")
    fake_wheel(wd, "winonly", "1.0", tag="py3-none-plan9_i386")
    return eb.Wheelhouse.scan(wd)


def names(chosen):
    return {n: str(w.version) for n, w in chosen.items()}


def test_chain_and_highest():
    with tempfile.TemporaryDirectory() as t:
        wh = make_house(Path(t))
        ch, notes = eb.mini_resolve(["starlette>=0.30", "pytest"], wh)
        got = names(ch)
        assert got["starlette"] == "0.50" and got["anyio"] == "4.4" and got["idna"] == "3.7"
        assert got["pytest"] == "8.0" and got["pluggy"] == "1.5" and got["iniconfig"] == "2.0"
        assert not [n for n in notes if "typing" in n]       # marker false -> never requested


def test_constraint_merge_picks_compatible_anyio():
    with tempfile.TemporaryDirectory() as t:
        wh = make_house(Path(t))
        ch, notes = eb.mini_resolve(["starlette==0.37", "anyio"], wh)   # starlette 0.37 needs anyio<5 only
        assert names(ch)["anyio"] == "4.4"
        ch, notes = eb.mini_resolve(["anyio<4", "starlette>=0.30"], wh)  # 0.50 needs anyio>=4 -> conflict
        assert any(n.startswith("unsatisfiable: anyio") for n in notes)  # no backtracking: reported, not hidden


def test_skip_missing_and_wrong_platform_and_self():
    with tempfile.TemporaryDirectory() as t:
        wh = make_house(Path(t))
        ch, notes = eb.mini_resolve(["httpbin", "winonly", "requests>=2", "ruff", "mkdocs-material"], wh,
                                    self_name="requests")
        assert ch == {} or set(ch) <= {"pytest"}
        assert any(n.startswith("no wheel: httpbin") for n in notes)
        assert any(n.startswith("no wheel: winonly") for n in notes)   # win wheel filtered by tag
        assert not any("requests" in n or "ruff" in n or "mkdocs" in n for n in notes)


def test_top_level_pytest_pin_ignored():
    with tempfile.TemporaryDirectory() as t:
        wh = make_house(Path(t))
        ch, notes = eb.mini_resolve(["pytest<=6.2.5"], wh)
        assert names(ch)["pytest"] == "8.0" and not notes


def test_extras_marker_and_unsatisfiable():
    with tempfile.TemporaryDirectory() as t:
        wh = make_house(Path(t))
        ch, notes = eb.mini_resolve(["pydantic"], wh)
        assert "annotated-types" in ch and "extra-only" not in ch and not any("extra-only" in n for n in notes)
        ch, notes = eb.mini_resolve(["pydantic[email]"], wh)
        assert any("extra-only" in n for n in notes)                    # requested extra -> looked up, no wheel
        ch, notes = eb.mini_resolve(["idna>=99"], wh)
        assert any(n.startswith("unsatisfiable: idna") for n in notes)


def test_requirement_list_formats():
    with tempfile.TemporaryDirectory() as t:
        r = Path(t) / "repo"; r.mkdir()
        (r / "pyproject.toml").write_text(
            '[project]\nname="x"\ndependencies=["starlette>=0.40,<0.50"]\n'
            '[project.optional-dependencies]\ntest=["pytest>=7"]\n'
            '[tool.poetry.dependencies]\npython="^3.8"\npygments="^2.13.0"\nmarkdown-it-py=">=2.2.0"\n'
            '[tool.poetry.group.dev.dependencies]\nattrs="^21.4.0"\n')
        (r / "requirements-tests.txt").write_text("-r requirements.txt\ntrustme==1.1.0 # c\n-e .\ngit+https://x/y\n")
        (r / "requirements.txt").write_text("httpx<1\n")
        got = eb.requirement_list(r)
        for want in ["starlette>=0.40,<0.50", "pytest>=7", "pygments>=2.13.0,<3", "markdown-it-py>=2.2.0",
                     "attrs>=21.4.0,<22", "trustme==1.1.0", "httpx<1"]:
            assert want in got, (want, got)
        assert got[-1] == "pytest" or "pytest" in got


def test_build_env_with_pip():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        wh = make_house(t)
        repo = t / "repo"; repo.mkdir()
        (repo / "requirements.txt").write_text("starlette>=0.30\n")
        dock = t / "docker"; dock.mkdir()
        (dock / "imp.py").write_text("# imp\n"); (dock / "telnetlib.py").write_text("# tn\n")
        info = eb.build_env(repo, t / "deps", wh, docker_dir=dock, repo_short="fastapi")
        assert info.n_wheels >= 5, info
        assert (t / "deps" / "starlette" / "__init__.py").is_file()
        assert (t / "deps" / "imp.py").is_file() and (t / "deps" / "telnetlib.py").is_file()
        assert not [n for n in info.notes if n.startswith("pip failed")], info.notes


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
