import numpy as np 
from itertools import product
from sklearn.base import clone
from sklearn.model_selection._search import ParameterGrid
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils.metaestimators import if_delegate_has_method
from joblib import Parallel, delayed

__all__ = ['NewGridSearchCV']

def _fit_and_score(estimator, X, y, dy, scorer, train_index, test_index, parameters):
    if dy is None:
        train_X, train_y, train_dy = X[train_index], y[train_index], None
        test_X, test_y, test_dy = X[test_index], y[test_index], None
    else:
        train_X, train_y, train_dy = X[train_index], y[train_index], dy[train_index]
        test_X, test_y, test_dy = X[test_index], y[test_index], dy[test_index]
    estimator.set_params(**parameters)
    estimator.fit(train_X, train_y, train_dy)
    if dy is None:
        test_score = scorer(estimator, test_X, test_y)
    elif hasattr(estimator, 'steps'):
        transformer = estimator.steps[0][1]     # only has one pca transformer now !
        test_dy_ = transformer.transform_gradient(test_dy)
        test_score = scorer(estimator, test_X, test_y, test_dy_)
    else:
        test_score = scorer(estimator, test_X, test_y, test_dy)
    n_test = len(test_index)
    return test_score, n_test

class NewGridSearchCV(object):
    def __init__(self, estimator, params_grid, scoring=None,\
                 cv=None, n_jobs=-1):
        """
        :estimator: estimator
        :fit_params: fitting parameter for the estimator
        :scoring: callable, sklearn scorer function
        :cv: integer or CV spliter, default use sklearn KFold spliter
        """
        self.estimator = estimator
        self.params_grid = params_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def _get_param_iterator(self):
        return ParameterGrid(self.params_grid)

    def _get_cv(self):
        cv = self.cv
        if isinstance(cv, BaseCrossValidator):
            return cv
        elif isinstance(cv, int):
            return KFold(cv)
        elif cv is None:
            return KFold()

    def fit(self, X, y, dy=None):
        base_estimator = clone(self.estimator)
        scorer = self.scoring
        cv = self._get_cv()
        n_splits = cv.get_n_splits()
        all_params_iter = self._get_param_iterator()
        all_params_list = list(all_params_iter)
        all_test_scores = []
        test_sample_counts = []
        for params in all_params_iter:
            test_scores = []
            for train_index, test_index in cv.split(X, y):
                scores, test_count = _fit_and_score(clone(base_estimator), X, y, dy,\
                                                scorer, train_index, test_index, params)
                test_scores.append(scores)
                test_sample_counts.append(test_count)
            all_test_scores.append(test_scores)
        all_test_scores = np.array(all_test_scores)
        # out = Parallel(n_jobs=self.n_jobs)(delayed(_fit_and_score)\
        #                                           (clone(base_estimator), X, y, dy, scorer,\
        #                                            train_index, test_index, params)
        #                                           for params, (train_index, test_index) in \
        #                                           product(all_params_iter, cv.split(X, y)))
        # out = list(zip(*out))
        # all_test_scores = np.array(out[0]).reshape((-1, n_splits))
        # test_sample_counts = out[1]
        # format output
        cv_results = {}
        for i in range(n_splits):
            cv_results['split%d_test_score' %(i)] = all_test_scores[:, i]
        mean_test_scores = np.average(all_test_scores, axis=1,\
                                                    weights=test_sample_counts[:n_splits])
        cv_results['mean_test_score'] = mean_test_scores
        cv_results['params'] = all_params_list
        self.cv_results_ = cv_results
        self.best_index_ = np.argmax(mean_test_scores)
        self.best_params_ = all_params_list[self.best_index_]
        self.best_score_ = np.amax(mean_test_scores)
        self.best_estimator_ = clone(base_estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(X, y, dy)
        return self

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        return self.best_estimator_.predict(X)
        
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_gradient(self, X):
        return self.best_estimator_.predict_gradient(X)