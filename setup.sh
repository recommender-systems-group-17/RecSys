#!/usr/bin/env bash
set -e
eval "$(conda shell.bash hook)"
conda create -n recsys python=3.10 -y
conda activate recsys
pip install -e .
pip install "numpy<2" "ray[tune]"
python -c "import recbole; print('RecBole', recbole.__version__, 'installed OK')"
