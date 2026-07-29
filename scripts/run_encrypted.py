import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from encrypted import BinaryRelevanceFHE, LabelPowersetClassifierFHE, _scores
from kdis_data import load_fold

METHODS = {"br": BinaryRelevanceFHE, "lp": LabelPowersetClassifierFHE}


def _run_fold(datasets_dir, dataset, method, fold, device, fhe):
    X_train, y_train, X_test, y_test = load_fold(datasets_dir, dataset, fold)
    model = METHODS[method](device=device, fhe=fhe)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.compile(X_train)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    scores = _scores(y_test, y_pred)
    return fold, scores, fit_time, compile_time, predict_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--output-dir", default="results/encrypted")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fhe", default="execute")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_run_fold, args.datasets_dir, args.dataset, args.method, fold, args.device, args.fhe)
                for fold in args.folds
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{args.dataset}/{args.method}] fold {result[0]}/10 done", flush=True)
    else:
        for fold in args.folds:
            result = _run_fold(args.datasets_dir, args.dataset, args.method, fold, args.device, args.fhe)
            results.append(result)
            print(f"[{args.dataset}/{args.method}] fold {result[0]}/10 done", flush=True)

    results.sort(key=lambda r: r[0])
    hamming_losses = [r[1]["hamming_loss"] for r in results]
    subset_accuracies = [r[1]["subset_accuracy"] for r in results]
    f1_micros = [r[1]["f1_micro"] for r in results]
    fit_times = [r[2] for r in results]
    compile_times = [r[3] for r in results]
    predict_times = [r[4] for r in results]

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
