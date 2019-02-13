import numpy as np 
from sklearn.base import clone
from sklearn.model_selection._search import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.utils.metaestimators import if_delegate_has_method

class NewGridSearchCV(object):
    def __init__(self, estimator, params_grid, scoring=None,\
                 iid=True, n_cv=3, n_jobs=None):
        """
        :estimator: estimator
        :fit_params: fitting parameter for the estimator
        :scoring: callable, sklearn scorer function
        :iid: distribution of data among different folds
        :cv: integer, number of fold 
        """
        self.estimator = estimator
        self.params_grid = params_grid
        self.scoring = scoring
        self.iid = iid
        self.n_cv = n_cv
        self.n_jobs = n_jobs

    def _get_param_iterator(self):
        return ParameterGrid(self.params_grid)
    
    def fit(self, X, y, dy=None):
        base_estimator = clone(self.estimator)
        cv = KFold(self.n_cv)
        all_params = self._get_param_iterator()
        all_params_list = list(all_params)
        all_test_scores = []
        for params in all_params:
            test_scores = []
            for train_index, test_index in cv.split(X):
                if dy is not None:
                    train_X, train_y, train_dy = X[train_index], y[train_index], dy[train_index]
                    test_X, test_y, test_dy = X[test_index], y[test_index], dy[test_index]
                else:
                    train_X, train_y, train_dy = X[train_index], y[train_index], None
                    test_X, test_y, test_dy = X[test_index], y[test_index], None
                estimator = clone(base_estimator)
                estimator.set_params(**params)
                estimator.fit(train_X, train_y, train_dy)
                score = self.scoring(estimator, test_X, test_y, test_dy)
                test_scores.append(score)
            all_test_scores.append(test_scores)
        all_test_scores = np.mean(np.array(all_test_scores), axis=1)
        self.cv_results_ = all_test_scores
        self.best_index_ = np.argmax(all_test_scores)
        self.best_params_ = all_params_list[self.best_index_]
        self.best_estimator_ = clone(base_estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(X, y, dy)
        return self

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        return self.best_estimator.predict(X)
        
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_gradient(self, X):
        return self.best_estimator.predict_gradient(X)