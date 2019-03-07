import numpy as np 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from scipy.optimize import minimize

from .utils import rbf_kernel
from ..ext_math import svd_solver, svd_inv

class GaussProcessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=1, beta=1.0, kernel=rbf_kernel, kernel_gd=None, optimize=False):
        self.gamma = gamma
        self.beta = beta
        self.kernel = kernel
        self.kernel_gd = kernel_gd
        self.optimize = optimize

    def neg_log_likelihood(self, hyperparams, X, y):
        alpha, inv_K, _ = self._fit(hyperparams, X, y)
        data_term = -0.5*(y-self._ymean).T @ alpha
        K_term = -0.5*np.log(1.0/np.linalg.det(inv_K))
        const_term = -0.5*self._n_dim*np.log(2*np.pi)
        return -(data_term + K_term + const_term)

    def neg_log_likelihood_prime(self, hyperparams, X, y):
        alpha, inv_K, K_gd_on_gamma = self._fit(hyperparams, X, y)
        reduce_K = np.outer(alpha, alpha) - inv_K 
        gd_on_gamma = 0.5*np.trace(reduce_K @ K_gd_on_gamma)
        gd_on_beta = 0.5*np.trace(reduce_K)
        return -np.array([gd_on_gamma, gd_on_beta])

    def _fit(self, hyperparams, X, y):
        gamma, beta = hyperparams
        K, K_gd_on_gamma = self.kernel(gamma, X, X, gradient_on_gamma=True)
        augment_K = K + beta*np.eye(self._n_dim)
        coef_ = svd_solver(augment_K, y-self._ymean)
        inv_K_ = svd_inv(augment_K)
        return coef_, inv_K_, K_gd_on_gamma

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._n_dim = len(X)
        self.X_fit_ = X
        self._ymean = np.mean(y)
        hyperparams = np.array([self.gamma, self.beta])
        if self.optimize:
            res = minimize(self.neg_log_likelihood, hyperparams, args=(X, y), method='L-BFGS-B',\
                           jac=self.neg_log_likelihood_prime, bounds=((0, 10), (0, 0.1)), options={'gtol': 1e-6, 'disp': True})
            print(res.success)
            hyperparams = res.x
        self.gamma, self.beta = hyperparams
        self.coef_, self.inv_cov_train_, _ = self._fit(hyperparams, X, y)
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
