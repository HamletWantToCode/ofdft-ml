# metric that accomodate the gradient estimation
import numpy as np 
from sklearn.metrics.scorer import _BaseScorer

class _PredictScorer(_BaseScorer):
    def __call__(self, estimator, X, y_true, dy_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_pred = estimator.predict(X)
        dyt_pred = estimator.predict_gradient(X)
        dy_pred = estimator.named_steps['reduce_dim'].inverse_transform_gradient(dyt_pred)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            y_err = self._sign * self._score_func(y_true, y_pred, **self._kwargs)
            dy_err = self._sign * self._score_func(np.ravel(dy_true), np.ravel(dy_pred), **self._kwargs)
            return y_err + dy_err

def make_scorer(score_func, greater_is_better=False, **kwargs):
    """Make a scorer from a performance metric or loss function.
    This factory function wraps scoring functions for use in GridSearchCV
    and cross_val_score. It takes a score function, such as ``accuracy_score``,
    ``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
    and returns a callable that scores an estimator's output.
    Read more in the :ref:`User Guide <scoring>`.
    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.
        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.
    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    sign = 1 if greater_is_better else -1
    return _PredictScorer(score_func, sign, kwargs)
