# new scorer

# metric that accomodate the gradient estimation
import numpy as np 
from sklearn.metrics.scorer import _BaseScorer, _ProbaScorer, _ThresholdScorer

class _PredictScorer(_BaseScorer):
    def __call__(self, estimator, X, y_true, dy_true=None):
        """
        Extend the function, allows to compute errors on function gradient if "dy_true" is provided
        """
        y_pred = estimator.predict(X)
        if dy_true is None:
            y_err = self._sign * self._score_func(y_true, y_pred, **self._kwargs)
            return y_err
        else:
            dy_pred = estimator.predict_gradient(X)
            y_err = self._sign * self._score_func(y_true, y_pred, **self._kwargs)
            dy_err = self._sign * self._score_func(np.ravel(dy_true), np.ravel(dy_pred), **self._kwargs)
            return y_err + dy_err

def make_scorer(score_func, greater_is_better=False, needs_proba=False,\
                needs_threshold=False, **kwargs):
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
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.
    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better."""

    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True,"
                         " but not both.")
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)
