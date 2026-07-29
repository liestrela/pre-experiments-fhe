import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
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


class BinaryRelevance:
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression()

    def fit(self, X, y):
        assert y.ndim == 2, "y must be a 2D array of shape (n_samples, n_labels)"
        self.n_labels_ = y.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)
        self.estimators_ = []
        for i in range(self.n_labels_):
            estimator = clone(self.base_estimator)
            estimator.fit(X, y[:, i])
            self.estimators_.append(estimator)
        return self

    def predict(self, X):
        X = self.scaler_.transform(X)
        columns = [estimator.predict(X) for estimator in self.estimators_]
        return np.column_stack(columns)

    def predict_proba(self, X):
        X = self.scaler_.transform(X)
        columns = [estimator.predict_proba(X)[:, 1] for estimator in self.estimators_]
        return np.column_stack(columns)

    def score(self, X, y):
        return _scores(y, self.predict(X))


class LabelPowersetClassifier:
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression()

    def fit(self, X, y):
        assert y.ndim == 2, "y must be a 2D array of shape (n_samples, n_labels)"
        self.lp_ = LabelPowerset()
        y_encoded = self.lp_.fit_transform(y)
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_encoded)
        return self

    def predict(self, X):
        X = self.scaler_.transform(X)
        y_encoded = self.estimator_.predict(X)
        return self.lp_.inverse_transform(y_encoded)

    def predict_proba(self, X):
        X = self.scaler_.transform(X)
        class_proba = self.estimator_.predict_proba(X)
        classes = self.estimator_.classes_
        label_proba = np.zeros((X.shape[0], self.lp_.n_labels))
        for col, class_label in enumerate(classes):
            combo = self.lp_.idx_to_combo[int(class_label)]
            for label, bit in enumerate(combo):
                if bit:
                    label_proba[:, label] += class_proba[:, col]
        return label_proba

    def score(self, X, y):
        return _scores(y, self.predict(X))
