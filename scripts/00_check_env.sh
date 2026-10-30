#!/bin/bash
#PBS -N de_check_env
#PBS -q regular
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:10:00
#PBS -e ../error/
#PBS -o ../output/

# T0.10: check that a Helix COMPUTE node has what the analysis needs (git, python3, an env manager,
# internet). Submit from the scripts/ folder of the repo:   cd darkedges/scripts && qsub 00_check_env.sh
# Create these folders once, at the repo root, before submitting:   mkdir -p log error output data
# Unlike a normal job, this one reports every check instead of stopping at the first missing tool.

set -Eeuo pipefail
cd "$PBS_O_WORKDIR"
DATA_DIR=../data
LOG=../log/log_${PBS_JOBID}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
handle_error() {
  local code=$?
  log "FAILED job=$PBS_JOBID host=$(hostname) line=$1 exit=$code time=$(date '+%F %T')"
  exit "$code"
}
trap 'handle_error $LINENO' ERR

# ---- EDIT PER JOB ----
MODULES=(anaconda3)                       # tried, not required: a missing module is only reported
CONDA_ENV=""                              # leave empty: this job checks the base setup
REQUIRED_CMDS=(git python3 timeout tar)   # must all exist for the check to pass
ENV_MANAGERS=(conda uv)                   # at least one must exist
URLS_REQUIRED=(https://pypi.org/simple/pip/)   # must be reachable from the compute node
URLS_INFO=(https://conda.anaconda.org/conda-forge/noarch/repodata.json.zst https://github.com)
# ----------------------

source /etc/profile.d/modules.sh 2>/dev/null || true
for m in "${MODULES[@]}"; do
  if module load "$m" 2>/dev/null; then log "loaded module: $m"; else log "module not available: $m"; fi
done

if [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  log "conda env: $CONDA_ENV"
fi

FAIL=0
log "host=$(hostname) jobid=$PBS_JOBID cores=$(nproc) queue=${PBS_QUEUE:-unknown}"

for cmd in "${REQUIRED_CMDS[@]}"; do
  if command -v "$cmd" >/dev/null; then
    log "OK   $cmd -> $(command -v "$cmd")"
  else
    log "MISS $cmd"; FAIL=1
  fi
done

FOUND_MANAGER=""
for cmd in "${ENV_MANAGERS[@]}"; do
  if command -v "$cmd" >/dev/null; then
    log "OK   env manager $cmd -> $(command -v "$cmd")"; FOUND_MANAGER=$cmd
  fi
done
if [[ -z "$FOUND_MANAGER" ]]; then log "MISS env manager (need one of: ${ENV_MANAGERS[*]})"; FAIL=1; fi

command -v git >/dev/null && log "git: $(git --version)"
if command -v python3 >/dev/null; then
  log "python3: $(python3 --version 2>&1)"
  for pkg in numpy scipy networkx sklearn pandas pyarrow yaml; do
    if python3 -c "import $pkg" 2>/dev/null; then log "py pkg present: $pkg"; else log "py pkg absent:  $pkg (environment.yml will provide it)"; fi
  done
fi

# Internet from this compute node: HTTP status via python3 (curl may be missing).
check_url() {
  timeout --foreground 30 python3 - "$1" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(urllib.request.Request(sys.argv[1], method="HEAD"), timeout=20) as r:
        print(r.status)
except Exception as e:
    print("ERR " + type(e).__name__)
    sys.exit(1)
PY
}
for u in "${URLS_REQUIRED[@]}"; do
  if out=$(check_url "$u" 2>/dev/null); then log "OK   internet $u -> $out"; else log "MISS internet $u -> ${out:-no answer}"; FAIL=1; fi
done
for u in "${URLS_INFO[@]}"; do
  if out=$(check_url "$u" 2>/dev/null); then log "info internet $u -> $out"; else log "info internet $u -> ${out:-no answer} (not required)"; fi
done

log "disk (here):"; df -h . | tee -a "$LOG"
if [[ -d "/scratch/projects/${USER:-unknown}" ]]; then log "scratch: /scratch/projects/$USER exists"; else log "scratch: /scratch/projects/${USER:-unknown} not found"; fi

if [[ $FAIL -eq 0 ]]; then
  log "T0.10 RESULT: ALL OK"
else
  log "T0.10 RESULT: NOT OK (see MISS lines above)"
  exit 1
fi