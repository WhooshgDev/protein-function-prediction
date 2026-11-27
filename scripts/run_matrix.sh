#!/usr/bin/env bash
# Final experiment matrix: E0-E3 x one seed (run several seeds in parallel shells).
# usage: bash scripts/run_matrix.sh <seed>    (seed 42 uses plain experiment names)
set -euo pipefail
cd "$(dirname "$0")/.."
seed=$1
for exp in ${EXPS:-baseline hierarchical taxonomy combined}; do
  name=$exp; [[ $seed != 42 ]] && name="${exp}_s${seed}"
  uv run python scripts/train.py --config configs/$exp.yaml --aspect all --name "$name" --set seed=$seed
  uv run python scripts/predict.py --config configs/$exp.yaml --input validation --name "$name"
  uv run python scripts/evaluate.py --config configs/$exp.yaml --predictions "predictions/${name}_val.tsv" --name "$name" --cafaeval-proteins $([[ $seed == 42 ]] && echo 3000 || echo 0)
done
