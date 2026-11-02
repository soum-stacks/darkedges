"""Per-task offline environment builder (the "mini-resolver" recipe, docs/env_recipe.md).

Stdlib + `packaging` (ships with pip). Python >= 3.11 (tomllib).

Recipe: collect requirements from the repo at base commit -> pick the highest compatible
wheel per requirement from the wheelhouse -> follow wheel METADATA (Requires-Dist) ->
skip anything with no wheel -> `pip install --no-deps --no-index --target <deps>` ->
copy docker/imp.py and docker/telnetlib.py into <deps>.
Difference from the notebook helpers: constraints on the same package are intersected
(fixed point) instead of first-come-first-served. Same result when there are no conflicts.
"""
from __future__ import annotations

import email.parser
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

# Tooling never needed to run tests (keeps setup fast, conflicts rare).
EXCLUDE = {"ruff", "mypy", "black", "isort", "flake8", "pylint", "pre-commit", "coverage",
           "codespell", "twine", "build", "pyright", "autoflake", "tox", "nox", "cairosvg",
           "pillow", "typer-cli"}
EXCLUDE_PREFIX = ("mkdocs", "types-", "sphinx", "flit", "hatch", "pytest-cov", "pytest-xdist",
                  "pytest-benchmark", "pytest-sugar", "setuptools-")
REPO_SELF = {"fastapi": "fastapi", "rich": "rich", "requests": "requests", "httpx": "httpx"}


def _excluded(name: str) -> bool:
    return name in EXCLUDE or name.startswith(EXCLUDE_PREFIX)


# ------------------------------------------------------------------ wheelhouse
@dataclass(frozen=True)
class Wheel:
    name: str
    version: Version
    path: Path


@dataclass
class Wheelhouse:
    by_name: dict = field(default_factory=dict)   # canonical name -> [Wheel], newest first
    _meta: dict = field(default_factory=dict)

    @classmethod
    def scan(cls, root) -> "Wheelhouse":
        ok = set(sys_tags())
        wh = cls()
        for p in sorted(Path(root).rglob("*.whl")):
            try:
                name, ver, _b, tags = parse_wheel_filename(p.name)
            except Exception:
                continue
            if set(tags) & ok:
                wh.by_name.setdefault(name, []).append(Wheel(name, ver, p))
        for lst in wh.by_name.values():
            lst.sort(key=lambda w: w.version, reverse=True)
        return wh

    def requires(self, w: Wheel) -> list:
        if w.path in self._meta:
            return self._meta[w.path]
        out = []
        try:
            with zipfile.ZipFile(w.path) as z:
                meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
                msg = email.parser.Parser().parsestr(z.read(meta).decode("utf-8", "replace"))
                out = msg.get_all("Requires-Dist") or []
        except Exception:
            pass
        self._meta[w.path] = out
        return out

    def pick(self, name: str, spec: SpecifierSet):
        cands = self.by_name.get(canonicalize_name(name), [])
        for pre in (False, True):          # prefer final releases
            for w in cands:
                if spec.contains(w.version, prereleases=pre):
                    return w
        return None


# ------------------------------------------------------------------ requirement collection
_REQ_FILES = ("requirements.txt", "requirements-tests.txt", "requirements-test.txt",
              "requirements-dev.txt", "requirements/tests.txt", "requirements/test.txt",
              "requirements/dev.txt", "tests/requirements.txt", "test-requirements.txt",
              "dev-requirements.txt")


def parse_requirements_text(text: str, base: Path | None = None, _depth: int = 0) -> list:
    out = []
    for raw in text.splitlines():
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")) and base is not None and _depth < 3:
            ref = base / line.split(None, 1)[1].strip()
            if ref.is_file():
                out += parse_requirements_text(ref.read_text(errors="replace"), ref.parent, _depth + 1)
            continue
        if line.startswith("-") or "://" in line or line.startswith((".", "/")):
            continue
        out.append(line)
    return out


def _poetry_to_req(name: str, val):
    if name.lower() == "python":
        return None
    if isinstance(val, dict):
        if {"git", "path", "url"} & set(val):
            return None
        val = val.get("version", "")
    if isinstance(val, list):
        return None
    v = str(val).strip()
    try:
        if v in ("", "*"):
            return name
        if v.startswith("^"):
            lo = Version(v[1:])
            hi = f"{lo.major + 1}" if lo.major else f"0.{lo.minor + 1}"
            return f"{name}>={lo},<{hi}"
        if v.startswith("~") and not v.startswith("~="):
            lo = Version(v[1:])
            return f"{name}>={lo},<{lo.major}.{lo.minor + 1}"
        return f"{name}=={v}" if v[0].isdigit() else f"{name}{v}"
    except Exception:
        return name


def requirement_list(repo_dir) -> list:
    """Plausible runtime + test requirement strings found in the repo (deduplicated, ordered)."""
    import tomllib
    repo = Path(repo_dir)
    reqs: list = []
    pp = repo / "pyproject.toml"
    if pp.is_file():
        try:
            data = tomllib.loads(pp.read_text(errors="replace"))
        except Exception:
            data = {}
        proj = data.get("project", {})
        reqs += proj.get("dependencies", [])
        for grp in proj.get("optional-dependencies", {}).values():
            reqs += grp
        for grp in data.get("dependency-groups", {}).values():
            reqs += [g for g in grp if isinstance(g, str)]
        flit = data.get("tool", {}).get("flit", {}).get("metadata", {})
        reqs += flit.get("requires", [])
        for grp in flit.get("requires-extra", {}).values():
            reqs += grp
        poetry = data.get("tool", {}).get("poetry", {})
        tables = [poetry.get("dependencies", {}), poetry.get("dev-dependencies", {})]
        tables += [g.get("dependencies", {}) for g in poetry.get("group", {}).values()]
        for t in tables:
            for n, v in t.items():
                r = _poetry_to_req(n, v)
                if r:
                    reqs.append(r)
    for rel in _REQ_FILES:
        f = repo / rel
        if f.is_file():
            reqs += parse_requirements_text(f.read_text(errors="replace"), f.parent)
    # setup.py (requests): best-effort, quoted specifiers with a comparison operator
    f = repo / "setup.py"
    if f.is_file():
        for m in re.finditer(r"""['"]([A-Za-z][A-Za-z0-9_.\-]*(?:\[[^\]]*\])?\s*[<>=!~]=?[^'"]*)['"]""",
                             f.read_text(errors="replace")):
            reqs.append(m.group(1).strip())
    reqs.append("pytest")
    seen, out = set(), []
    for r in reqs:
        r = r.strip()
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ------------------------------------------------------------------ resolver
def mini_resolve(reqs, wh: Wheelhouse, self_name: str | None = None, max_iter: int = 25):
    """-> (chosen {canonical name: Wheel}, notes [str]). Constraints on a name are intersected."""
    notes: list = []
    cons: dict = {}       # name -> SpecifierSet
    extras: dict = {}     # name -> set(extras)
    self_c = canonicalize_name(self_name) if self_name else None

    def add(req_str: str, check_marker: bool) -> bool:
        try:
            r = Requirement(req_str)
        except InvalidRequirement:
            notes.append(f"unparsable: {req_str}")
            return False
        n = canonicalize_name(r.name)
        if n == self_c or _excluded(n):
            return False
        if check_marker and r.marker is not None and not r.marker.evaluate({"extra": ""}):
            return False
        new = cons.get(n, SpecifierSet()) & r.specifier
        ex = extras.setdefault(n, set())
        changed = n not in cons or str(new) != str(cons[n]) or not set(r.extras) <= ex
        cons[n] = new
        ex |= set(r.extras)
        return changed

    for s in reqs:
        add(s, True)

    chosen: dict = {}
    expanded: dict = {}   # name -> (wheel path, frozenset(extras)) already expanded
    for _ in range(max_iter):
        changed = False
        for n in list(cons):
            w = wh.pick(n, cons[n])
            if w is None:
                msg = (f"unsatisfiable: {n}{cons[n]}" if n in wh.by_name else f"no wheel: {n}")
                if msg not in notes:
                    notes.append(msg)
                chosen.pop(n, None)
                expanded.pop(n, None)
                continue
            chosen[n] = w
            key = (w.path, frozenset(extras[n]))
            if expanded.get(n) == key:
                continue
            expanded[n] = key
            for rd in wh.requires(w):
                try:
                    r = Requirement(rd)
                except InvalidRequirement:
                    continue
                if r.marker is not None and not any(
                        r.marker.evaluate({"extra": e}) for e in ("", *extras[n])):
                    continue
                r.marker = None
                if add(str(r), False):
                    changed = True
            changed = True
        if not changed:
            break
    # drop "no wheel" notes for names that ended up with a wheel
    return chosen, notes


# ------------------------------------------------------------------ build
@dataclass
class EnvInfo:
    deps_dir: Path
    n_wheels: int
    notes: list
    seconds: float


def build_env(repo_dir, deps_dir, wheelhouse: Wheelhouse, docker_dir=None, repo_short=None,
              python=None, timeout=300) -> EnvInfo:
    t0 = time.time()
    deps = Path(deps_dir)
    if deps.exists():
        shutil.rmtree(deps)
    deps.mkdir(parents=True)
    chosen, notes = mini_resolve(requirement_list(repo_dir), wheelhouse,
                                 self_name=REPO_SELF.get(repo_short or "", repo_short))
    paths = sorted({str(w.path) for w in chosen.values()})
    if paths:
        cmd = [python or sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
               "--no-compile", "--disable-pip-version-check", "--quiet",
               "--target", str(deps), *paths]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            notes.append("pip failed: " + (p.stderr or p.stdout)[-300:].replace("\n", " "))
    for fn in ("imp.py", "telnetlib.py"):
        src = Path(docker_dir) / fn if docker_dir else None
        if src and src.is_file():
            shutil.copy2(src, deps / fn)
        else:
            notes.append(f"missing docker/{fn}")
    return EnvInfo(deps, len(paths), notes, round(time.time() - t0, 2))
