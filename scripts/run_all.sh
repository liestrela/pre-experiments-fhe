#!/usr/bin/env bash
set -euo pipefail

DATASETS=(flags emotions VirusGO GpositivePseAAC PlantPseAAC yeast HumanPseAAC)
METHODS=(br lp)

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "=== plaintext $dataset $method ==="
    uv run scripts/run_plaintext.py --dataset "$dataset" --method "$method"

    echo "=== encrypted $dataset $method ==="
    uv run scripts/run_encrypted.py --dataset "$dataset" --method "$method"
  done
done
