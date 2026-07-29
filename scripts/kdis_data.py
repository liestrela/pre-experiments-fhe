import tarfile
from pathlib import Path

import pandas as pd

DATASETS = ["flags", "emotions", "VirusGO", "GpositivePseAAC", "PlantPseAAC", "yeast", "HumanPseAAC"]


def load_fold(datasets_dir, dataset, fold):
    tar_path = Path(datasets_dir) / f"{dataset}.tar.gz"
    with tarfile.open(tar_path) as tar:
        labels_df = pd.read_csv(tar.extractfile(f"{dataset}/CrossValidation/namesLabels.csv"))
        n_labels = labels_df.shape[0]
        train_df = pd.read_csv(tar.extractfile(f"{dataset}/CrossValidation/Tr/{dataset}-Split-Tr-{fold}.csv"))
        test_df = pd.read_csv(tar.extractfile(f"{dataset}/CrossValidation/Ts/{dataset}-Split-Ts-{fold}.csv"))

    X_train = train_df.iloc[:, :-n_labels].apply(pd.to_numeric).to_numpy(dtype=float)
    y_train = train_df.iloc[:, -n_labels:].apply(pd.to_numeric).to_numpy(dtype=int)
    X_test = test_df.iloc[:, :-n_labels].apply(pd.to_numeric).to_numpy(dtype=float)
    y_test = test_df.iloc[:, -n_labels:].apply(pd.to_numeric).to_numpy(dtype=int)

    return X_train, y_train, X_test, y_test
