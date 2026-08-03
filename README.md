# Multi-label classification over encrypted data

Two implementations of multi-label classification (MLC): `plaintext.py` (scikit-learn) and `encrypted.py` (Concrete ML, fully homomorphic encryption). Both support:

- Binary Relevance (BR): one binary logistic regression classifier per label
- Label Powerset (LP): a single multiclass classifier over the unique label combinations seen in training

The encrypted classifiers follow a fit → compile → predict workflow, with FHE mode (`disable`/`simulate`/`execute`) and device (`cpu`/`cuda`) configurable per model.

## Getting started

Managed with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Requires Python >=3.10,<3.13. CUDA wheels for `concrete-python` are pulled from Zama's package index (configured in `pyproject.toml`); CPU-only works out of the box.

## Usage

Follows scikit-learn's nomenclature: instantiate, then `fit`/`predict`. `predict_proba` and `score` (Hamming loss, subset accuracy, micro-F1) are also available.

```python
from plaintext import BinaryRelevance, LabelPowersetClassifier

model = BinaryRelevance()  # or LabelPowersetClassifier()
model.fit(train_x, train_y)
predictions = model.predict(test_x)
scores = model.score(test_x, test_y)
```

```python
from encrypted import BinaryRelevanceFHE, LabelPowersetClassifierFHE

model = BinaryRelevanceFHE(device="cuda", fhe="execute")  # or LabelPowersetClassifierFHE(...)
model.fit(train_x, train_y)
model.compile(train_x)
predictions = model.predict(test_x)
scores = model.score(test_x, test_y)
```

## Running experiments

`scripts/` runs both methods, in both settings, over seven KDIS datasets (Flags, Emotions, VirusGO, GpositivePseAAC, PlantPseAAC, Yeast, HumanPseAAC) provided as 10-fold CV splits under `datasets/`.

```
scripts/run_plaintext.py --dataset <name> --method {br,lp}
scripts/run_encrypted.py --dataset <name> --method {br,lp} --device cuda --fhe execute
scripts/run_all.sh
```

Each run writes a summary CSV (mean ± std across folds) to `results/plaintext/` or `results/encrypted/`; `run_all.sh` skips combinations already present. `scripts/generate_tables.py` turns `results/` into LaTeX tables under `tables/tables.tex`.
