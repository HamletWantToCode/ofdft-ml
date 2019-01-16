# Gauss Process Regression
import numpy as np 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y

from .utils import rbf_kernel
from ext_math import svd_inv

class GaussProcess(BaseEstimator, RegressorMixin):
    """ 
    Parameters
    ----------
    sigma:
    gamma:
    kernel:
    gradient_on:
    kernel_gd:
    kernel_hess:

    Attributes
    ----------
    X_fit_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_fit_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    dy_fit_ : ndarray, shape (n_classes,)
    """
    def __init__(self, sigma=1e-5, gamma=1e-3, kernel=rbf_kernel, kernel_gd=None, gradient_on=False, kernel_hess=None):
        self.sigma = sigma
        self.gamma = gamma
        self.kernel = kernel
        self.gradient_on = gradient_on
        self.kernel_gd = kernel_gd
        self.kernel_hess = kernel_hess

    def fit(self, X, y, dy=None):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        dy: array-like, shape (n_samples, n_features)

        Returns
        -------
        self : object
            Returns self.
        """
        X, y= check_X_y(X, y)
        if dy is None:
            dy = np.zeros_like(X)
        dy = check_array(dy)
        assert dy.shape[0] == X.shape[0], print('dimension of dy and X mismatch !')
        N_tr, D = X.shape
        dy = np.ravel(dy)
        self.X_fit_ = X
        self.y_fit_ = y
        self.dy_fit_ = dy

        A = self.kernel(self.gamma, X, X) + (self.sigma**2)*np.eye(N_tr)
        if self.gradient_on:
            C = self.kernel_gd(self.gamma, X, X).reshape((N_tr*D, N_tr))
            B = C.T
            D = self.kernel_hess(self.gamma, X, X) + (self.sigma**2)*np.eye(N_tr*D)
        else:
            B = np.zeros((N_tr, N_tr*D))
            C = B.T
            D = np.zeros((N_tr*D, N_tr*D))
        D_inv = svd_inv(D)
        self.ul_block_ = svd_inv((A - B @ D_inv @ C))
        self.ur_block_ = -self.ul_block_ @ B @ D_inv
        self.bl_block_ = -D_inv @ C @ self.ul_block_
        self.br_block_ = D_inv + D_inv @ C @ self.ul_block_ @ B @ D_inv
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
        """
        X = check_array(X)
        N_ts = X.shape[0]
        N_tr, D = self.X_fit_.shape
        A = self.kernel(self.gamma, X, self.X_fit_)
        if self.gradient_on:
            B = self.kernel_gd(self.gamma, self.X_fit_, X).reshape((N_tr*D, N_ts)).T
        else:
            B = np.zeros((N_ts, N_tr*D))
        pred_y = A @ (self.ul_block_ @ self.y_fit_ + self.ur_block_ @ self.dy_fit_) + B @ (self.bl_block_ @ self.y_fit_ + self.br_block_ @ self.dy_fit_)
        return pred_y

    def predict_gradient(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        dy : ndarray, shape (n_samples, n_features)
        """
        X = check_array(X)
        N_ts, D = X.shape
        N_tr = self.X_fit_.shape[0]
        A = self.kernel_gd(self.gamma, X, self.X_fit_).reshape((N_ts*D, N_tr))
        if self.gradient_on:
            B = self.kernel_hess(self.gamma, X, self.X_fit_)
        else:
            B = np.zeros((N_ts*D, N_tr*D))
        pred_dy = A @ (self.ul_block_ @ self.y_fit_ + self.ur_block_ @ self.dy_fit_) + B @ (self.bl_block_ @ self.y_fit_ + self.br_block_ @ self.dy_fit_)
        return pred_dy.reshape((N_ts, D))

    def predict_variance(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        var : ndarray, shape (n_samples,)
        """
        X = check_array(X)
        N_ts, D = X.shape
        N_tr = self.X_fit_.shape[0]
        K = self.kernel(self.gamma, X, X)
        A = self.kernel(self.gamma, X, self.X_fit_)
        if self.gradient_on:
            B = self.kernel_gd(self.gamma, self.X_fit_, X).reshape((N_tr*D, N_ts)).T
        else:
            B = np.zeros((N_ts, N_tr*D))
        covariance = K - (A @ (self.ul_block_ @ A.T + self.ur_block_ @ B.T) + B @ (self.bl_block_ @ A.T +self.br_block_ @ B.T))
        return np.sqrt(np.diag(covariance))

