#!/usr/bin/env bash
# Run from the repo root on the laptop (Git Bash/WSL) after committing. Produces darkedges-code.zip
# containing src/ (tracelens + darkedges), skills/ (TraceLens agent skill) and configs/agent_check_tasks.txt.
# Upload it as a new version of the PRIVATE Kaggle dataset "darkedges-code".
set -euo pipefail
paths=(src skills)
git ls-files --error-unmatch configs/agent_check_tasks.txt >/dev/null 2>&1 && paths+=(configs/agent_check_tasks.txt)
git archive --format=zip -o darkedges-code.zip HEAD "${paths[@]}"
echo "wrote darkedges-code.zip ($(du -h darkedges-code.zip | cut -f1)) with: ${paths[*]}"
