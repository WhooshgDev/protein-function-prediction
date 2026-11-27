#!/usr/bin/env bash
# After scripts/run_matrix.sh {42,43,44}: test exports, diagnostics, error analysis, tables.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="uv run python"
for exp in ${FINAL_EXPS:-baseline hierarchical taxonomy combined}; do
  $PY scripts/predict.py --config configs/$exp.yaml --input testsuperset
  $PY scripts/error_analysis.py --config configs/$exp.yaml --predictions predictions/${exp}_val.tsv
done
rm -f results/hierarchy_diagnostics.json
$PY scripts/hierarchy_diagnostics.py --names baseline hierarchical taxonomy combined hier_l0.01 hier_l0.1 hier_l1.0 hier_l10 hier_l100
$PY scripts/compare_experiments.py --seeds 42 43 44 \
  --sweep baseline hier_l0.01 hier_l0.1 hier_l1.0 hier_l10 hier_l100 hier_l1000
