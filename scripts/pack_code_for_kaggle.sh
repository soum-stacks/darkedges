#!/usr/bin/env bash
# Run from the repo root on the laptop (Git Bash/WSL) after committing. Produces darkedges-code.zip
# containing src/ only (tracelens + darkedges). Upload it as a PRIVATE Kaggle dataset "darkedges-code".
set -euo pipefail
git archive --format=zip -o darkedges-code.zip HEAD src
echo "wrote darkedges-code.zip ($(du -h darkedges-code.zip | cut -f1))"
