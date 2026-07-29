#!/usr/bin/env bash
set -euo pipefail

DATASETS=(flags emotions VirusGO GpositivePseAAC PlantPseAAC yeast HumanPseAAC)
METHODS=(br lp)
PLAINTEXT_WORKERS="${PLAINTEXT_WORKERS:-1}"

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "=== plaintext $dataset $method ==="
    uv run scripts/run_plaintext.py --dataset "$dataset" --method "$method" --workers "$PLAINTEXT_WORKERS"

    echo "=== encrypted $dataset $method ==="
    uv run scripts/run_encrypted.py --dataset "$dataset" --method "$method" --workers "1"
  done
done
