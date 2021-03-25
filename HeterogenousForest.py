import numpy as np

from joblib import Parallel, delayed

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample

class HeterogenousForest():

    def __init__(self, base, n_estimators, max_depth, bootstrap = True, n_jobs = 1, *args, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.args = args
        self.kwargs = kwargs
        self.base = base
        self.bootstrap = bootstrap
        
        self.n_jobs = n_jobs
        # Do I still need these?
        self.verbose = False

        assert max_depth > 0 and max_depth is not None

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
        scaled_prob = np.array([w * p for w,p in zip(all_proba, self.weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(proba.argmax(axis=1), axis=0)

    def _fit(self, X, y):
        d = np.random.randint(1, self.max_depth)
        n_samples = X.shape[0]
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        
        if self.bootstrap:
            #X, y = resample(X, y)
            # This is pretty much taken from _parallel_build_trees in scikit-learns ensemble/_forest.py 
            indices = np.random.randint(0, n_samples, n_samples)
            sample_counts = np.bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts

        #tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)

        model = self.base(max_depth = d, *self.args, **self.kwargs)
        model.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
        # print(model.classes_)
        return model

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(unique_labels(y))
        # print(self.classes_)
        
        self.X_ = X
        self.y_ = y

        self.estimators_ = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self._fit) (X, y) for _ in range(self.n_estimators)
        )