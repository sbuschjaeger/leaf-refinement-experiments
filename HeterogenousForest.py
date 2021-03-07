import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

class HeterogenousForest():

    def __init__(self, base, n_estimators, depth = [], *args, **kwargs):
        self.n_estimators = n_estimators
        self.depth = depth
        self.args = args
        self.kwargs = kwargs
        self.base = base

    def _individual_proba(self, X):
        assert self.estimators_ is not None, "Call fit before calling predict_proba!"
        all_proba = []

        for e in self.estimators_:
            tmp = np.zeros(shape=(X.shape[0], self.n_classes_), dtype=np.float32)
            tmp[:, e.classes_.astype(int)] += e.predict_proba(X)
            all_proba.append(tmp)

        if len(all_proba) == 0:
            return np.zeros(shape=(1, X.shape[0], self.n_classes_), dtype=np.float32)
        else:
            return np.array(all_proba)

    def predict_proba(self, X):
        all_proba = self._individual_proba(X)
        combined_proba = np.mean(all_proba, axis=0)
        return combined_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(unique_labels(y))

        self.X_ = X
        self.y_ = y

        n_est_per_base = self.n_estimators // len(self.depth) + (self.n_estimators % len(self.depth) > 0)
        self.estimators_ = []
        for d in self.depth:
            model = self.base(n_estimators = n_est_per_base, max_depth = d, *self.args, **self.kwargs)
            model.fit(X, y)
            self.n_jobs = model.n_jobs
            self.verbose = model.verbose
            self.estimators_.extend(model.estimators_)
        self.n_estimators = len(self.estimators_)
