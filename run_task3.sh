#!/usr/bin/env bash
# Reproduces all of Task 3 from the shared Task 1 outputs (~6 min).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY task3/sweep.py                          # 3.1 + 3.2
$PY task3/pipelines.py                      # 3.3
$PY task3/groups.py                         # 3.4 (figure for EASE)
$PY task3/groups.py --model Hybrid-Weighted
