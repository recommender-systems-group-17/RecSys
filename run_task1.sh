#!/usr/bin/env bash
# Runs all of Task 1: tune + train individual models, then build the hybrids.
#   bash run_task1.sh            full run
#   bash run_task1.sh --quick    smoke test (1 config per model, 2 epochs)
set -e
python task1/individual.py "$@"
python task1/hybrids.py
