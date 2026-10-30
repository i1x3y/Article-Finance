#!/usr/bin/env bash
# Reproduces every number, table and figure in the paper. Run from the repository root.
set -euo pipefail
mkdir -p results paper
echo "[1/8] Benchmarks, econometric, ML (default) and hybrid models"; python pipeline.py 12 36 60
echo "[2/8] Tuned ML models";      for w in 12 36 60; do python tuned.py $w 0 $w; done
echo "[3/8] Deep learning models";  for w in 12 36 60; do for i in $(seq 0 $(( w/12 - 1 ))); do python dl_chunk.py $w $i; done; done
echo "[4/8] Merge DL forecasts";    python merge.py
echo "[5/8] Final evaluation (RMSE, DM-HLN, MCS)"; python analysis.py eval
echo "[6/8] Sub-periods, descriptives, importance"; python analysis.py sub; python analysis.py desc; python analysis.py imp
echo "[7/8] MASE, directional accuracy, selected hyperparameters"; python extra_metrics.py; python select_params.py 60 48 36 24 12; python make_tables_extra.py
echo "[8/8] Tables and figures";   python make_outputs.py
echo "Done. Results in results/, tables and figures in paper/."
