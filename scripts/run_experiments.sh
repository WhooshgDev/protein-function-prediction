#!/usr/bin/env bash
# Runs the experiment matrix used for results/REPORT.md (assumes steps 1-2 are done).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="uv run python"
train_eval () {  # $1 config, $2 name, rest: extra --set overrides
  local cfg=$1 name=$2; shift 2
  $PY scripts/train.py --config "$cfg" --aspect all --name "$name" "$@"
  $PY scripts/predict.py --config "$cfg" --input validation --name "$name" "$@"
  $PY scripts/evaluate.py --config "$cfg" --predictions "predictions/${name}_val.tsv" --name "$name" "$@"
}
stage=${1:-all}
if [[ $stage == sweep || $stage == all ]]; then
  for lam in ${LAMBDAS:-0.01 0.1 1.0 10 100 1000}; do
    train_eval configs/hierarchical.yaml "hier_l${lam}" --set loss.hierarchical_lambda=$lam
  done
fi
if [[ $stage == main || $stage == all ]]; then
  train_eval configs/hierarchical.yaml hierarchical
  train_eval configs/taxonomy.yaml taxonomy
  train_eval configs/combined.yaml combined
fi
if [[ $stage == seeds || $stage == all ]]; then
  for seed in 43 44; do
    for exp in baseline hierarchical taxonomy combined; do
      $PY scripts/train.py --config configs/$exp.yaml --aspect all --name "${exp}_s${seed}" --set seed=$seed
    done
  done
fi
