import numpy as np 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y

from .utils import rbf_kernel
from ..ext_math import svd_solver, svd_inv

class GaussProcessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=1, beta=0.1, kernel=rbf_kernel, kernel_gd=None):
        self.gamma = gamma
        self.beta = beta
        self.kernel = kernel
        self.kernel_gd = kernel_gd

    def fit(self, X, y, dy=None):
        X, y = check_X_y(X, y)
        n_dim = len(X)
        self.X_fit_ = X
        self._ymean = np.mean(y)
        cov_train = self.kernel(self.gamma, X, X)
        augment_cov_train = cov_train + self.beta*np.eye(n_dim)
        self.coef_ = svd_solver(augment_cov_train, y-self._ymean)
        self.inv_cov_train_ = svd_inv(augment_cov_train)
        return self

    def predict(self, X, return_variance=False):
        X = check_array(X)
        kT = self.kernel(self.gamma, X, self.X_fit_)
        y_pred = self._ymean + kT @ self.coef_
        cov_pred = self.kernel(self.gamma, X, X) - kT @ self.inv_cov_train_ @ kT.T
        if return_variance:
            return y_pred, np.sqrt(np.diag(cov_pred))
        else:
            return y_pred

    def predict_gradient(self, X):
        assert self.kernel_gd is not None, print('compute gradient need kernel gradient !')
        X = check_array(X)
        dy_pred = self.kernel_gd(self.gamma, X, self.X_fit_) @ self.coef_
        return dy_pred
