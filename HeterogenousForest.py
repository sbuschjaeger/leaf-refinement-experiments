import numpy as np

from joblib import Parallel, delayed

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier

# class PoissonDecisionTreeClassifier():
#     def __init__(self, max_depth, *args, **kwargs):
#         self.model = None
#         self.max_depth = max_depth
#         self.args = args
#         self.kwargs = kwargs
    
#     def predict_proba(self, X):
#         assert self.model is not None, "Class fit before calling predict_proba!"
#         return self.model.predict_proba(X)

#     def predict(self, X):
#         assert self.model is not None, "Class fit before calling predict"
#         return self.model.predict(X)

#     def fit(self, X, y):
#         d = np.random.poisson(self.max_depth - 1) + 1
#         self.model = DecisionTreeClassifier(max_depth=d, *self.args, **self.kwargs)
#         return self.model.fit(X,y)

class PoissonDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth, subspace_features,
        criterion="gini",
        splitter="best",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        class_weight=None,
        ccp_alpha=0.0
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)

        self.subspace_features = subspace_features

    def predict_proba(self, X):
        return super().predict_proba(X[:, self.features])

    def predict(self, X):
        return super().predict(X[:, self.features])

    def fit(self, X, y, sample_weight=None):
        if self.max_depth is not None:
            d = np.random.poisson(self.max_depth - 1) + 1
            self.max_depth = d
        
        if self.subspace_features < 1.0:
            n_features = int(X.shape[1] * self.subspace_features)
            self.features = np.random.choice(range(X.shape[1]), n_features, replace=False)
        else:
            self.features = range(X.shape[1])
        super().fit(X[:, self.features], y, sample_weight)

# Implement this as random patches!
class HeterogenousForest():

    def __init__(self, n_estimators, max_depth, max_samples = 1.0, bootstrap = True, n_jobs = 1, *args, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.args = args
        self.kwargs = kwargs
        self.bootstrap = bootstrap
        self.max_samples = max_samples

        self.n_jobs = n_jobs
        # Do I still need these?
        self.verbose = False

        assert max_depth > 1 and max_depth is not None
        assert max_samples <= 1 and max_samples > 0

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
        #d = np.random.randint(1, self.max_depth)
        d = np.random.poisson(self.max_depth - 1) + 1
        n_samples = X.shape[0]
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        
        if self.bootstrap:
            #X, y = resample(X, y)
            # This is pretty much taken from _parallel_build_trees in scikit-learns ensemble/_forest.py 
            indices = np.random.randint(0, n_samples, int(self.max_samples * n_samples))

            sample_counts = np.bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts

        #tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)

        model = DecisionTreeClassifier(max_depth = d, *self.args, **self.kwargs)
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
        self.weights_ = [1.0/self.n_estimators for _ in range(self.n_estimators)]

        self.estimators_ = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self._fit) (X, y) for _ in range(self.n_estimators)
        )