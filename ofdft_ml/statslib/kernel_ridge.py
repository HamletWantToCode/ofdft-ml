# KRR

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array

from .utils import rbf_kernel
from ..ext_math import svd_solver

class KernelRidge(BaseEstimator, RegressorMixin):
    """
    Nonlinear regression method with translational invariant kernel & L_2 regularization

    :param C: float, small positive values to improve the conditioning of the problem

    :param gamma: float, small value of gamma lead to a large correlation length in RBF kernel

    :param kernel: callable

    :param kernel_gd: callable

    Examples::

    >>> from statslib.kernel_ridge import KernelRidge
    >>> import numpy as np
    >>> X = np.random.rand(10, 5)
    >>> y = np.sum(X**2) + np.sin(X[0])
    >>> model = KernelRidge()
    >>> model.fit(X[:6], y[:6])
    >>> y_hat = model.predict(X[-4:])
    >>> dy_hat = model.predict_gradient(X[-4:])
    """
    def __init__(self, C=1e-10, gamma=1e-3, kernel=rbf_kernel, kernel_gd=None):
        self.gamma = gamma
        self.C = C
        self.kernel = kernel
        self.kernel_gd = kernel_gd

    def fit(self, X, y, dy=None):
        """
        Build kernel matrix from data, and solve for model coefficients.

        :param X: array-like, shape (n_samples, n_features), the training input samples.

        :param y: array-like, shape (n_samples,), the target values.

        :param dy: None, left for consistency

        :return self: object, returns self.
        """
        X, y = check_X_y(X, y)
        self.X_fit_ = X
        n_dims = X.shape[0]
        A = self.kernel(self.gamma, X, X) + self.C*n_dims*np.eye(n_dims)
        self.cond_ = np.linalg.cond(A)
        self.coef_ = svd_solver(A, y)
        return self

    def predict(self, X):
        """
        predict the function values for newly input samples.

        :param X: array-like, shape (n_samples, n_features), the input samples.

        :return y_pred: ndarray, shape (n_samples,)
        """
        X = check_array(X)
        y_pred = self.kernel(self.gamma, X, self.X_fit_) @ self.coef_
        return y_pred

    def predict_gradient(self, X):
        """
        compute the gradient based on the fitted function

        :param X: array-like, shape (n_samples, n_features), the input samples.

        :return dy_pred : ndarray, shape (n_samples, n_features)
        """
        assert self.kernel_gd is not None, print('need to specify gradient of kernel !')
        X = check_array(X)
        n_samples, n_features = X.shape
        dy_pred = self.kernel_gd(self.gamma, X, self.X_fit_) @ self.coef_
        return dy_pred.reshape((n_features, n_samples)).T
