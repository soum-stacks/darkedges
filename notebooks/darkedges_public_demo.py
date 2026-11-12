# %% [markdown]
# # Dark Edges: public reproduction notebook (no competition data)
#
# Companion to the Paper Track writeup **"Dark Edges: The Gemma 4 code graphs miss method calls, not functions, and filling
# them in does not fix localisation"**. Code: https://github.com/soum-stacks/darkedges-paper (Apache-2.0).
#
# This notebook uses **no competition data** (the rules forbid redistributing it). It shows:
# 1. the TraceLens tracer on a tiny example package, and what a *dark edge* is;
# 2. the unit tests of the analysis pipeline (synthetic data);
# 3. the paper's repair and ladder figures redrawn from the released aggregate results.
#
# Settings: Internet **On** (to clone the public repo). CPU is enough; runs in about 2 minutes.

# %%
import subprocess, sys, os, json, textwrap
REPO = "/kaggle/working/darkedges-paper"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "main",
                    "https://github.com/soum-stacks/darkedges-paper.git", REPO], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO], check=True)
print(sys.version)
print(open(f"{REPO}/README.md").read()[:1500])

# %% [markdown]
# ## 1. A dark edge in 20 lines
# A static call graph built from names (like the competition graphs) links `report()` to `fmt()`, but not
# `Table.render()` to `Table._width()`: `self._width()` only says that *some* `_width` method runs.
# TraceLens records the calls that really happen.

# %%
demo = "/kaggle/working/demo"
os.makedirs(f"{demo}/pkg", exist_ok=True)
open(f"{demo}/pkg/__init__.py", "w").write("")
open(f"{demo}/pkg/core.py", "w").write(textwrap.dedent('''
    def fmt(s, w):
        return str(s).ljust(w)

    class Table:
        def __init__(self, rows):
            self.rows = rows
        def render(self):
            return [fmt(r, self._width()) for r in self.rows]
        def _width(self):
            return max(len(str(r)) for r in self.rows)

    def report(rows):
        return fmt("n=%d" % len(rows), 6) + str(Table(rows).render())
'''))
open(f"{demo}/run.py", "w").write("from pkg.core import report\nprint(report(['a', 'bbb']))\n")
subprocess.run([sys.executable, "-m", "tracelens.run", "--root", demo, "--out", f"{demo}/trace.json", "--", f"{demo}/run.py"],
               cwd=demo, check=True)
tr = json.load(open(f"{demo}/trace.json"))
nodes = {n["i"]: n["qualname"] for n in tr["nodes"]}
runtime = {(nodes[e["caller"]], nodes[e["callee"]]) for e in tr["edges"] if e.get("caller") is not None}

# a name-based static graph: only plain-name calls f(...) resolve; self.m() does not
import ast
src = ast.parse(open(f"{demo}/pkg/core.py").read())
static = set()
for top in src.body:
    funcs = [top] if isinstance(top, ast.FunctionDef) else [(top, m) for m in top.body if isinstance(m, ast.FunctionDef)]
    for item in funcs:
        cls, fn = (None, item) if not isinstance(item, tuple) else item
        caller = fn.name if cls is None else f"{cls.name}.{fn.name}"
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                static.add((caller, c.func.id))
print(f"{'runtime call':35s} in static graph?")
for a, b in sorted(runtime):
    if a == "<module>":
        continue
    print(f"{a + ' -> ' + b:35s} {'yes' if (a, b) in static else 'NO  <- dark edge'}")

# %% [markdown]
# ## 2. Unit tests (synthetic data)
# `test_envbuild` and `test_tracing` need the Kaggle competition wheelhouse, so they are skipped here.

# %%
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                    "--ignore=tests/test_envbuild.py", "--ignore=tests/test_tracing.py"],
                   cwd=REPO, capture_output=True, text=True)
print(r.stdout[-2500:], r.stderr[-1000:])

# %% [markdown]
# ## 3. Figures from the released aggregate results
# `paper/results/*.json` hold aggregates only (no task data). The full pipeline that produced them is in `scripts/`
# (it needs the competition data; see the README).

# %%
import matplotlib.pyplot as plt
rep = json.load(open(f"{REPO}/paper/results/repair_summary.json"))
lad = json.load(open(f"{REPO}/paper/results/ladder_summary.json"))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
steps = [("orig", "released"), ("R1", "+self calls"), ("R2", "+attributes"), ("R3", "+constructors")]
for repo, col in zip(["fastapi", "rich", "requests"], ["#2a78d6", "#eb6834", "#1baf7a"]):
    if repo in rep["dark_fraction"]:
        ys = [rep["dark_fraction"][repo][k]["mean"] for k, _ in steps]
        ax[0].plot(range(4), ys, "-o", color=col, label=f"{repo} (n={rep['dark_fraction'][repo]['n_tasks']})")
ax[0].set_xticks(range(4), [l for _, l in steps]); ax[0].set_ylim(0, 1)
ax[0].set_ylabel("dark fraction"); ax[0].set_title("Graph repair (paper Fig. 3)", loc="left"); ax[0].legend(frameon=False)

order = ["R0_bm25", "R3_tests_idf", "R4b_bm25_in_tests", "R4_hybrid_rrf", "R5_oracle_ochiai"]
names = ["BM25 (issue)", "existing tests", "BM25 in tests' fns", "hybrid", "oracle Ochiai"]
vals = [lad["pooled"][m]["20"]["mean"] for m in order]
ax[1].barh(range(5), vals, color=["#1baf7a", "#1baf7a", "#2a78d6", "#2a78d6", "#bbbbbb"])
ax[1].axvline(lad["coverage_ceiling"]["mean"], ls="--", color="#eb6834")
ax[1].set_yticks(range(5), names); ax[1].invert_yaxis(); ax[1].set_xlim(0, 1)
ax[1].set_xlabel("recall@20"); ax[1].set_title("Non-oracle ladder (paper Fig. 4)", loc="left")
plt.tight_layout(); plt.show()
print("gap closed at K=20:", lad["gap_closed_at_20"])
for f in ("repair_summary.md", "ladder_summary.md", "agent_check_summary.md"):
    p = f"{REPO}/paper/results/{f}"
    if os.path.exists(p):
        print("=" * 30, f); print(open(p).read()[:2500])
