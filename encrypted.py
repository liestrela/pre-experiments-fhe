import numpy as np
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.preprocessing import StandardScaler


class LabelPowerset:
    def __init__(self):
        self.combo_to_idx = {}
        self.idx_to_combo = {}
        self.n_labels = None

    def fit_transform(self, y):
        self.n_labels = y.shape[1]
        combos = [tuple(row) for row in y]
        unique_combos = sorted(set(combos))
        self.combo_to_idx = {c: i for i, c in enumerate(unique_combos)}
        self.idx_to_combo = {i: c for c, i in self.combo_to_idx.items()}
        return np.array([self.combo_to_idx[c] for c in combos], dtype=np.int32)

    def inverse_transform(self, y_encoded):
        rows = [self.idx_to_combo[int(i)] for i in y_encoded]
        return np.array(rows, dtype=np.int32)

    @property
    def n_classes(self):
        return len(self.combo_to_idx)


def _scores(y_true, y_pred):
    return {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
    }


def _predict_fhe(estimator, X, fhe):
    if fhe == "execute":
        n = len(X)
        y = np.empty(n, dtype=np.int32)
        for j in range(n):
            y[j] = estimator.predict(X[j:j + 1], fhe="execute")[0]
            print(f"  {j + 1}/{n} ({(j + 1) / n * 100:.0f}%)", end="\r", flush=True)
        print()
        return y
    return estimator.predict(X, fhe=fhe)


def _predict_proba_fhe(estimator, X, fhe):
    if fhe == "execute":
        n = len(X)
        probas = []
        for j in range(n):
            probas.append(estimator.predict_proba(X[j:j + 1], fhe="execute")[0])
            print(f"  {j + 1}/{n} ({(j + 1) / n * 100:.0f}%)", end="\r", flush=True)
        print()
        return np.array(probas)
    return estimator.predict_proba(X, fhe=fhe)


class BinaryRelevanceFHE:
    def __init__(self, base_estimator_cls=None, base_estimator_kwargs=None, device="cuda", fhe="execute"):
        self.base_estimator_cls = base_estimator_cls if base_estimator_cls is not None else ConcreteLogisticRegression
        self.base_estimator_kwargs = base_estimator_kwargs if base_estimator_kwargs is not None else {}
        self.device = device
        self.fhe = fhe

    def fit(self, X, y):
        assert y.ndim == 2, "y must be a 2D array of shape (n_samples, n_labels)"
        self.n_labels_ = y.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)
        self.estimators_ = []
        for i in range(self.n_labels_):
            estimator = self.base_estimator_cls(**self.base_estimator_kwargs)
            estimator.fit(X, y[:, i])
            self.estimators_.append(estimator)
        return self

    def compile(self, X):
        X = self.scaler_.transform(X)
        for estimator in self.estimators_:
            estimator.compile(X, device=self.device)
        return self

    def predict(self, X, fhe=None):
        fhe = fhe if fhe is not None else self.fhe
        X = self.scaler_.transform(X)
        columns = [_predict_fhe(estimator, X, fhe) for estimator in self.estimators_]
        return np.column_stack(columns)

    def predict_proba(self, X, fhe=None):
        fhe = fhe if fhe is not None else self.fhe
        X = self.scaler_.transform(X)
        columns = [_predict_proba_fhe(estimator, X, fhe)[:, 1] for estimator in self.estimators_]
        return np.column_stack(columns)

    def score(self, X, y, fhe=None):
        return _scores(y, self.predict(X, fhe=fhe))


class LabelPowersetClassifierFHE:
    def __init__(self, base_estimator_cls=None, base_estimator_kwargs=None, device="cuda", fhe="execute"):
        self.base_estimator_cls = base_estimator_cls if base_estimator_cls is not None else ConcreteLogisticRegression
        self.base_estimator_kwargs = base_estimator_kwargs if base_estimator_kwargs is not None else {}
        self.device = device
        self.fhe = fhe

    def fit(self, X, y):
        assert y.ndim == 2, "y must be a 2D array of shape (n_samples, n_labels)"
        self.lp_ = LabelPowerset()
        y_encoded = self.lp_.fit_transform(y)
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)
        self.estimator_ = self.base_estimator_cls(**self.base_estimator_kwargs)
        self.estimator_.fit(X, y_encoded)
        return self

    def compile(self, X):
        X = self.scaler_.transform(X)
        self.estimator_.compile(X, device=self.device)
        return self

    def predict(self, X, fhe=None):
        fhe = fhe if fhe is not None else self.fhe
        X = self.scaler_.transform(X)
        y_encoded = _predict_fhe(self.estimator_, X, fhe)
        return self.lp_.inverse_transform(y_encoded)

    def predict_proba(self, X, fhe=None):
        fhe = fhe if fhe is not None else self.fhe
        X = self.scaler_.transform(X)
        class_proba = _predict_proba_fhe(self.estimator_, X, fhe)
        classes = self.estimator_.classes_
        label_proba = np.zeros((X.shape[0], self.lp_.n_labels))
        for col, class_label in enumerate(classes):
            combo = self.lp_.idx_to_combo[int(class_label)]
            for label, bit in enumerate(combo):
                if bit:
                    label_proba[:, label] += class_proba[:, col]
        return label_proba

    def score(self, X, y, fhe=None):
        return _scores(y, self.predict(X, fhe=fhe))
