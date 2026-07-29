import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encrypted import BinaryRelevanceFHE
from kdis_data import DATASETS, load_fold

CALIBRATION_DATASET = "flags"
N_CALIBRATION_SAMPLES = 20
N_FOLDS = 10


def main():
    X_train, y_train, X_test, y_test = load_fold("datasets", CALIBRATION_DATASET, 1)

    model = BinaryRelevanceFHE(device="cuda", fhe="execute")
    model.fit(X_train, y_train)
    model.compile(X_train)

    n = min(N_CALIBRATION_SAMPLES, X_test.shape[0])
    t0 = time.perf_counter()
    model.predict(X_test[:n])
    elapsed = time.perf_counter() - t0

    per_execution = elapsed / (n * y_train.shape[1])
    print(f"measured: {per_execution * 1000:.1f} ms per (sample, label) real FHE execution")

    total_executions = 0
    for dataset in DATASETS:
        _, y_tr, X_ts, _ = load_fold("datasets", dataset, 1)
        n_labels = y_tr.shape[1]
        test_rows = X_ts.shape[0]
        total_executions += test_rows * n_labels * N_FOLDS
        total_executions += test_rows * N_FOLDS

    projected_seconds = total_executions * per_execution
    print(
        f"projected total suite runtime (execute-only, excludes fit/compile): "
        f"~{projected_seconds / 3600:.1f} hours across {total_executions} executions"
    )


if __name__ == "__main__":
    main()
