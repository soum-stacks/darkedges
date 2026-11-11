# %% [markdown]
# # T3.3 Agent check (H4): E4B x {no_graph_tools, static_graph_tools, tracelens} x 30 tasks x 2 seeds
#
# **Where:** Kaggle notebook, Accelerator **GPU L4 x4**, Internet **Off**.
# **Add Input:** (1) the MAIN competition `gemma-4-developer-agent`; (2) dataset
# `metric/gemma-4-developer-agent-wheelhouse`; (3) model `google/gemma-4` -> `gemma-4-e4b-it-qat-w4a16-ct`
# (version 2); (4) your PRIVATE dataset `darkedges-code` (new version that contains `src/`, `skills/` and
# `configs/agent_check_tasks.txt`, built by `scripts/pack_code_for_kaggle.sh`).
# **Run All once**, do not restart the vLLM server. Resumable: re-running skips finished (task, condition, seed).
# **Output:** `/kaggle/working/agent_check_outputs.zip` (agent_check.jsonl + per-run patches/logs; PRIVATE,
# competition-derived). Download it from the Output panel and copy to Helix `results/`.
# Budget is the frozen protocol budget (configs/protocol.yaml agent_check). Seed 0 for all conditions runs first.

# %%
import glob, importlib, json, logging, os, re, shutil, subprocess, sys, time
from pathlib import Path

T_START = time.time()
_gpus = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout
print(sys.version.split()[0], "|", _gpus.strip().replace("\n", " | ") or "NO GPU")
assert sys.version_info[:2] == (3, 12), (
    f"Python {sys.version.split()[0]}: the wheelhouse is cp312. Use Accelerator = GPU L4 x4 on the starter's "
    "environment (Copy & Edit the official starter notebook), then restart the session.")
assert _gpus.count("L4") >= 4, "Accelerator must be GPU L4 x4 (attach the main competition to make it selectable)."
DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
WORK = Path("/kaggle/working")
OUT = WORK / "agent_check"
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "agent_check.jsonl"
assert (DATA_DIR / "tasks.jsonl").exists(), "Attach the MAIN competition (Add Input -> Competitions)."

for k, v in {"TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_FLAX": "1", "TRANSFORMERS_OFFLINE": "1",
             "VLLM_NO_USAGE_STATS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
             "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "NCCL_DEBUG": "WARN", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
             "LITELLM_TELEMETRY": "False", "LITELLM_LOG": "WARNING", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
             "OTEL_SDK_DISABLED": "true", "OTEL_PYTHON_DISABLED": "true"}.items():
    os.environ[k] = v
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "EMPTY")
os.environ["LOCAL_API_KEY"] = os.environ.get("LOCAL_API_KEY", "EMPTY")


def quiet():
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx", "httpcore", "openai", "urllib3",
                 "opentelemetry", "opentelemetry.context"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        if "litellm" in name.lower():
            lg.propagate = False
            lg.handlers.clear()


quiet()
WHEELHOUSE_DIR = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
if not WHEELHOUSE_DIR.exists():
    hits = glob.glob("/kaggle/input/**/gemma-4-developer-agent-wheelhouse", recursive=True)
    WHEELHOUSE_DIR = Path(hits[0]) if hits else WHEELHOUSE_DIR
assert WHEELHOUSE_DIR.exists(), "Attach dataset metric/gemma-4-developer-agent-wheelhouse"
for pat in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth", "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
    for pth in glob.glob(pat):
        try:
            os.unlink(pth)
        except OSError:
            pass
tmp_whl = Path("/tmp/wheelhouse")
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob("*.whl"):
    if "cutlass" in w.name.lower():
        continue
    tn = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
    if not (tmp_whl / tn).exists():
        try:
            os.symlink(w, tmp_whl / tn)
        except OSError:
            shutil.copyfile(w, tmp_whl / tn)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                *sorted(str(w) for w in tmp_whl.glob("*.whl"))], check=True)
importlib.invalidate_caches()
quiet()
print("wheelhouse installed;", sys.version.split()[0])

# %%
# ---------------- configuration (frozen protocol values; change only CONCURRENCY / SEEDS if quota is short)
MODEL_NAME = "gemma-4-e4b-it-qat-w4a16-ct"
CONDITIONS = ["no_graph_tools", "static_graph_tools", "tracelens"]
SEEDS = [0, 1]
BUDGET = {"timeout_seconds": 100, "max_tool_calls": 50, "max_time_minutes": 10, "max_turns": 60}
CONCURRENCY = 4          # tasks evaluated in parallel inside one condition; set 1 if anything misbehaves
SMOKE_ONLY = False       # True: run 1 task in the tracelens condition and stop (check the tool works)

code_hits = glob.glob("/kaggle/input/**/src/tracelens/tracer.py", recursive=True)
assert code_hits, "Attach your private dataset darkedges-code (new version with src/ and skills/)."
CODE = Path(code_hits[0]).parents[2]
print("code dataset:", CODE)
ids_file = CODE / "configs" / "agent_check_tasks.txt"
if ids_file.exists():
    TASK_IDS = [x.strip() for x in ids_file.read_text().splitlines() if x.strip()]
else:
    TASK_IDS = []  # paste the TASK_IDS list printed by scripts/11_select_agent_tasks.py here
assert len(TASK_IDS) == 30, f"expected 30 task ids, got {len(TASK_IDS)}"

# TraceLens skill on a host path the agent can call with run_command (subprocess sandbox shares the host FS)
TL = WORK / "tl"
shutil.rmtree(TL, ignore_errors=True)
shutil.copytree(CODE / "skills" / "tracelens", TL)
shutil.copytree(CODE / "src" / "tracelens", TL / "scripts" / "tracelens")
TRACE_CMD = f"python3 {TL}/scripts/trace_run.py"

# self-test of the tracer in this interpreter (synthetic repo, no competition data)
st = Path("/tmp/tl_selftest")
shutil.rmtree(st, ignore_errors=True)
(st / "pkg").mkdir(parents=True)
(st / "tests").mkdir()
(st / "pkg" / "__init__.py").write_text("")
(st / "pkg" / "core.py").write_text(
    "class T:\n    def run(self):\n        return self._w() + 1\n    def _w(self):\n        raise ValueError('bug')\n")
(st / "tests" / "_repro.py").write_text("from pkg.core import T\nT().run()\n")
p = subprocess.run(f"cd {st} && {TRACE_CMD} tests/_repro.py", shell=True, capture_output=True, text=True)
print(p.stdout[-1500:], p.stderr[-500:])
assert "T._w" in p.stdout and "SUSPECTS" in p.stdout, "tracer self-test failed"

# %%
# ---------------- agent directories: one per (condition, seed); identical except tools / localisation step
import yaml

class _IncLoader(yaml.SafeLoader):
    pass


_IncLoader.add_constructor("!include", lambda loader, node: {"__include__": loader.construct_scalar(node)})
SAMPLE = DATA_DIR / "sample_submission"
print((SAMPLE / "agent.yaml").read_text(encoding="utf-8"))  # recorded for the methods section
sample_cfg = yaml.load((SAMPLE / "agent.yaml").read_text(encoding="utf-8"), Loader=_IncLoader) or {}
GEN = sample_cfg.get("generate_content_config") or {}
if isinstance(GEN, dict) and "__include__" in GEN:
    GEN = yaml.load((SAMPLE / GEN["__include__"]).read_text(encoding="utf-8"), Loader=_IncLoader) or {}
GEN = dict(GEN) if isinstance(GEN, dict) else {}
GEN.pop("seed", None)
if not GEN:
    GEN = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_output_tokens": 16384,
           "thinking_config": {"thinking_budget": 4096, "include_thoughts": True}}
print("generate_content_config (from sample, seed added per run):", GEN)

BASE_TOOLS = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"]
GRAPH_TOOLS = ["get_code_neighbors", "get_code_subgraph", "search_similar_code"]
LOCALISE = {
    "no_graph_tools": "Use run_command (grep -rn) and read_file to find the function to change.",
    "static_graph_tools": ("Use get_code_neighbors, get_code_subgraph and search_similar_code on symbols named in "
                           "the issue, together with grep and read_file, to find the function to change."),
    "tracelens": ("First run the tracer on your reproduction with run_command: " + TRACE_CMD + " tests/_repro.py "
                  "It prints SUSPECTS: the repository functions that really ran, failure stack first, with file:line, "
                  "and the runtime calls into methods that the code graph lacks. Read the top suspects first. "
                  "You may also trace an existing test: " + TRACE_CMD + " -m pytest tests/test_x.py -q -x "
                  "The graph tools get_code_neighbors, get_code_subgraph and search_similar_code are also available."),
}


def instruction(cond: str) -> str:
    s = ("You are an expert software engineer fixing an issue in the Python repository at /workspace. "
         "Work in this order.\n"
         "1. Reproduce: write a short script at tests/_repro.py with write_file. It imports the package and fails "
         "(assertion or exception) because of the issue. Run it: python3 tests/_repro.py\n"
         "2. Localise: " + LOCALISE[cond] + "\n"
         "3. Fix the library code with small edit_file edits. Never edit existing tests, pytest.ini or conftest.py.\n"
         "4. Verify: run tests/_repro.py again and, if quick, the most relevant existing test file.\n"
         "5. Call submit_patch. Files under tests/ are removed from the patch automatically.\n"
         "Keep reasoning short and act with tools.")
    assert "{" not in s and "}" not in s, "ADK would treat braces as state templates"
    return s


AGENT_DIRS = {}
for cond in CONDITIONS:
    for seed in SEEDS:
        d = OUT / "agents" / f"{cond}_s{seed}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        cfg = {"name": "coder", "model": MODEL_NAME, "instruction": instruction(cond),
               "tools": BASE_TOOLS + ([] if cond == "no_graph_tools" else GRAPH_TOOLS),
               "generate_content_config": {**GEN, "seed": seed}}
        (d / "agent.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, width=1000), encoding="utf-8")
        (d / "eval_config.yaml").write_text(yaml.safe_dump({"evaluation": BUDGET}), encoding="utf-8")
        AGENT_DIRS[(cond, seed)] = d
print((AGENT_DIRS[("tracelens", 0)] / "agent.yaml").read_text())

# %%
# ---------------- vLLM server (start once; about 6 minutes)
from swegemma.models import load_tasks          # import swegemma before litellm (starter order; avoids a circular import)
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

litellm.drop_params = True
MODEL_PATH = Path(f"/kaggle/input/models/google/gemma-4/other/{MODEL_NAME}/2")
assert MODEL_PATH.exists(), f"Attach the Gemma 4 E4B model input; expected {MODEL_PATH}"
adapters = discover_adapters(str(AGENT_DIRS[("tracelens", 0)]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
tp = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
vcfg = VllmConfig(model=str(MODEL_PATH), port=8000, host="127.0.0.1", tool_call_parser="gemma4",
                  reasoning_parser="gemma4", max_model_len=32768,
                  dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto",
                  gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=False, max_loras=1,
                  max_lora_rank=128, scheduler_reserve_full_isl=False, tensor_parallel_size=tp,
                  startup_timeout=60 * 20, extra_args=())
if "server_instance" not in globals() or server_instance is None:
    server_instance = VllmServer(vcfg, adapter_manifest=adapters)
    t0 = time.time()
    server_instance.start()
    print(f"vLLM up in {time.time() - t0:.0f} s (tp={tp})")
models = server_instance.create_model_registry(aliases=[MODEL_NAME], model_prefix="openai/",
                                               api_key=os.environ["LOCAL_API_KEY"])

# %%
# ---------------- runner
import asyncio, concurrent.futures
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

TASKS_PATH = DATA_DIR / "tasks.jsonl"
TASKS = {t.instance_id: t for t in load_tasks(TASKS_PATH)}
RAW = {}
for line in TASKS_PATH.read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        RAW[r["instance_id"]] = r
missing = [i for i in TASK_IDS if i not in TASKS]
assert not missing, f"unknown ids: {missing}"
limits, gen_constraints = build_submission_limits()
DIFF_FILE = re.compile(r"^(?:diff --git a/(\S+) b/\S+|\+\+\+ b/(\S+)|--- a/(\S+))", re.M)


def patch_files(text):
    """Changed file paths from a unified diff (git or plain ---/+++ headers)."""
    out = set()
    for m in DIFF_FILE.finditer(text or ""):
        f = next((g for g in m.groups() if g), None)
        if f and f != "dev/null":
            out.add(f)
    return out


def is_test_file(p):
    parts = p.split("/")
    return parts[-1].startswith("test_") or parts[-1].endswith("_test.py") or parts[-1] == "conftest.py" or any(
        x in ("tests", "test", "testing") for x in parts[:-1])


def gold_files(iid):
    text = RAW[iid].get("patch") or RAW[iid].get("gold_patch") or getattr(TASKS[iid], "patch", "") or ""
    return {f for f in patch_files(text) if not is_test_file(f)}


_ng = {i: len(gold_files(i)) for i in TASK_IDS}
print("gold files per task:", _ng)
assert all(_ng.values()), f"gold patch files not parsed for: {[i for i, n in _ng.items() if not n]}"


def make_evaluator(agent_dir, results_dir):
    return Evaluator(EvalConfig(
        tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / "snapshots", results_dir=results_dir,
        submission_dir=agent_dir, models=models, sandbox="subprocess",
        timeout_seconds=BUDGET["timeout_seconds"], verification_timeout_seconds=300,
        max_time_minutes=float(BUDGET["max_time_minutes"]), max_tool_calls=BUDGET["max_tool_calls"],
        max_turns=BUDGET["max_turns"], limits=limits, generation_constraints=gen_constraints,
        adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                        token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA_DIR / "graphs"), embeddings_dir=str(DATA_DIR / "embeddings"),
        wheels_dir=DATA_DIR / "wheels", verbose=False))


def trace_text(results_dir, iid):
    parts = []
    for p in (results_dir / "traces" / f"trace_{iid}.json", results_dir / "logs" / f"{iid}.log"):
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def row_for(cond, seed, iid, r, results_dir, secs_wall, err=None):
    patch = (getattr(r, "agent_patch", "") or "").strip() if r is not None else ""
    pf = {f for f in patch_files(patch) if not is_test_file(f)}
    gf = gold_files(iid)
    txt = trace_text(results_dir, iid)
    return {
        "instance_id": iid, "repo": RAW[iid]["repo"], "condition": cond, "seed": seed,
        "resolved": bool(getattr(r, "resolved", False)) if r is not None else False,
        "test_exit_code": getattr(r, "test_exit_code", None) if r is not None else None,
        "patch_chars": len(patch), "n_patch_files": len(pf), "n_gold_files": len(gf),
        "gold_file_touched": bool(pf & gf),
        "tool_calls": getattr(r, "tool_calls", None) if r is not None else None,
        "seconds": round(float(getattr(r, "duration_seconds", 0) or 0), 1) if r is not None else None,
        "wall_seconds": round(secs_wall, 1),
        # approximate counts from the run transcript (string matches; see docs/t3_choices.md)
        "trace_ok": txt.count("TRACELENS exit_code="), "trace_mentions": txt.count("trace_run.py"),
        "graph_mentions": sum(txt.count(t) for t in GRAPH_TOOLS),
        "error_responses": txt.count("'error_type'") + txt.count('"error_type"'),
        "malformed_calls": txt.count("ToolNotFound") + txt.count("not found in tools") + txt.count("cut off"),
        "error": err or (str(getattr(r, "error", "") or "")[:300] if r is not None else None),
        "model": MODEL_NAME, "budget": BUDGET, "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def done_keys():
    keys = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if not (d.get("error") or "").startswith("EXC:"):
                    keys.add((d["instance_id"], d["condition"], d["seed"]))
    return keys


async def run_batch(cond, seed, ids):
    results_dir = OUT / "runs" / f"{cond}_s{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    ev = make_evaluator(AGENT_DIRS[(cond, seed)], results_dir)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(k, iid):
        async with sem:
            t0 = time.time()
            try:
                r = await ev.evaluate_task(task=TASKS[iid], task_index=k, total_tasks=len(ids))
                row = row_for(cond, seed, iid, r, results_dir, time.time() - t0)
            except Exception as e:  # recorded, retried on the next Run All
                row = row_for(cond, seed, iid, None, results_dir, time.time() - t0, err=f"EXC: {type(e).__name__}: {e}"[:300])
            with open(JSONL, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{cond} s{seed}] {iid}: resolved={row['resolved']} gold_file={row['gold_file_touched']} "
                  f"calls={row['tool_calls']} trace_ok={row['trace_ok']} {row['wall_seconds']}s {row['error'] or ''}",
                  flush=True)

    await asyncio.gather(*(one(k, iid) for k, iid in enumerate(ids, 1)))


def run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


# %%
# ---------------- smoke: 1 task, tracelens condition. Check trace_ok >= 1 in the printed line.
smoke_id = TASK_IDS[0]
if (smoke_id, "tracelens", 0) not in done_keys():
    run_sync(run_batch("tracelens", 0, [smoke_id]))
last = [json.loads(x) for x in JSONL.read_text().splitlines() if x.strip()][-1]
print(json.dumps(last, indent=1))
if last["trace_ok"] == 0:
    print("WARNING: the agent did not get tracer output. Log tail follows; check that it called the tracer.")
    print(trace_text(OUT / "runs" / "tracelens_s0", smoke_id)[-4000:])
assert not SMOKE_ONLY, "SMOKE_ONLY=True: stopping here"

# %%
# ---------------- full run: seed-major, so seed 0 is complete for all conditions before seed 1 starts
# DO NOT interrupt this cell. The batch runs in a worker thread that keeps going after an interrupt; re-running the
# cell then starts a second, concurrent copy (this happened in run 1). If interrupted: Run -> Restart kernel, then
# re-run cells 1-5 and this cell (finished runs are skipped). Preferred: Save Version -> Save & Run All.
import threading
assert threading.active_count() < 50, "stray worker threads alive: restart the kernel before re-running"
for seed in SEEDS:
    for cond in CONDITIONS:
        todo = [i for i in TASK_IDS if (i, cond, seed) not in done_keys()]
        print(f"=== {cond} seed {seed}: {len(todo)} to run ({(time.time() - T_START) / 3600:.2f} h elapsed)")
        if todo:
            run_sync(run_batch(cond, seed, todo))

# %%
# ---------------- summary + package (PRIVATE outputs)
import importlib.metadata
import pandas as pd

rows = {}
for line in JSONL.read_text().splitlines():
    if line.strip():
        d = json.loads(line)
        rows[(d["instance_id"], d["condition"], d["seed"])] = d  # last row per key wins
df = pd.DataFrame(rows.values())
print(df.groupby("condition")[["resolved", "gold_file_touched", "tool_calls", "seconds", "trace_ok"]].mean().round(3))
print(df.groupby(["condition", "seed"]).size())
meta = {"model": MODEL_NAME, "budget": BUDGET, "conditions": CONDITIONS, "seeds": SEEDS, "concurrency": CONCURRENCY,
        "generate_content_config": GEN, "python": sys.version.split()[0], "wall_hours": round((time.time() - T_START) / 3600, 3)}
for pkg in ("swegemma", "adk-submission", "google-adk", "vllm"):
    try:
        meta[pkg] = importlib.metadata.version(pkg)
    except Exception:
        pass
(OUT / "run_meta.json").write_text(json.dumps(meta, indent=1))
for c in CONDITIONS:
    shutil.copy(AGENT_DIRS[(c, 0)] / "agent.yaml", OUT / f"agent_{c}.yaml")
zp = shutil.make_archive(str(WORK / "agent_check_outputs"), "zip", root_dir=OUT)
print("wrote", zp, f"{Path(zp).stat().st_size / 1e6:.1f} MB; wall {meta['wall_hours']} h")
try:
    server_instance.stop()
except Exception as e:
    print("stop:", e)
