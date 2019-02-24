import numpy as np 
from itertools import product
from sklearn.base import clone
from sklearn.model_selection._search import ParameterGrid
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils.metaestimators import if_delegate_has_method

__all__ = ['NewGridSearchCV']

def _fit_and_score(estimator, X, y, dy, scorer, cv, parameters):
    cv_test_scores = []
    test_sample_counts = []
    for (train_index, test_index) in cv.split(X, y):
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
        elif hasattr(estimator, 'steps'):          # if we reduce the dimension of the space, we need to transform the gradient in original space to the new space
            transformer = estimator.steps[0][1]       # only has one pca transformer now !
            test_dy_ = transformer.transform_gradient(test_dy)
            test_score = scorer(estimator, test_X, test_y, test_dy_)
        else:
            test_score = scorer(estimator, test_X, test_y, test_dy)
        cv_test_scores.append(test_score)
        test_sample_counts.append(len(test_index))
    mean_cv_score = np.average(cv_test_scores, weights=test_sample_counts)
    return mean_cv_score

class NewGridSearchCV(object):
    def __init__(self, estimator, params_grid, scoring=None, cv=None):
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
        model_params_iter = self._get_param_iterator()
        all_results = []
        # grid search loop
        for params in model_params_iter:
            score_ = _fit_and_score(clone(base_estimator), X, y, dy, scorer, cv, params)
            all_results.append((score_, params))
        all_mean_scores, all_params_list = zip(*all_results)
        # format output
        cv_results = {}
        cv_results['mean_test_score'] = all_mean_scores
        cv_results['params'] = all_params_list
        self.cv_results_ = cv_results
        self.best_index_ = np.argmax(all_mean_scores)
        self.best_params_ = all_params_list[self.best_index_]
        self.best_score_ = np.amax(all_mean_scores)
        self.best_estimator_ = clone(base_estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(X, y, dy)
        return self

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        return self.best_estimator_.predict(X)
        
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_gradient(self, X):
        return self.best_estimator_.predict_gradient(X)