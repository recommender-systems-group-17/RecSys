#!/usr/bin/env bash
# Reproduces all of Task 2 from the shared Task 1 outputs (~1 min).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY task2/evaluate.py
$PY task2/groups.py
$PY task2/hybrid_analysis.py
