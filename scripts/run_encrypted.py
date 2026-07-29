import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from encrypted import BinaryRelevanceFHE, LabelPowersetClassifierFHE, _scores
from kdis_data import load_fold

METHODS = {"br": BinaryRelevanceFHE, "lp": LabelPowersetClassifierFHE}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--output-dir", default="results/encrypted")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fhe", default="execute")
    args = parser.parse_args()

    hamming_losses, subset_accuracies, f1_micros = [], [], []
    fit_times, compile_times, predict_times = [], [], []

    for fold in args.folds:
        X_train, y_train, X_test, y_test = load_fold(args.datasets_dir, args.dataset, fold)
        model = METHODS[args.method](device=args.device, fhe=args.fhe)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        model.compile(X_train)
        compile_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_times.append(time.perf_counter() - t0)

        scores = _scores(y_test, y_pred)
        hamming_losses.append(scores["hamming_loss"])
        subset_accuracies.append(scores["subset_accuracy"])
        f1_micros.append(scores["f1_micro"])

        print(f"[{args.dataset}/{args.method}] fold {fold}/10 done", flush=True)

    row = {
        "dataset": args.dataset,
        "method": args.method,
        "device": args.device,
        "fhe": args.fhe,
        "n_folds": len(args.folds),
        "hamming_loss_mean": np.mean(hamming_losses),
        "hamming_loss_std": np.std(hamming_losses),
        "subset_accuracy_mean": np.mean(subset_accuracies),
        "subset_accuracy_std": np.std(subset_accuracies),
        "f1_micro_mean": np.mean(f1_micros),
        "f1_micro_std": np.std(f1_micros),
        "fit_time_s_mean": np.mean(fit_times),
        "fit_time_s_std": np.std(fit_times),
        "compile_time_s_mean": np.mean(compile_times),
        "compile_time_s_std": np.std(compile_times),
        "predict_time_s_mean": np.mean(predict_times),
        "predict_time_s_std": np.std(predict_times),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_{args.method}.csv"
    pd.DataFrame([row]).to_csv(output_path, index=False)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
