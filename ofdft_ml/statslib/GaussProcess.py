import numpy as np 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from scipy.optimize import minimize
from scipy.linalg import cho_solve

from .utils import rbf_kernel

__all__ = ['GaussProcessRegressor']

class GaussProcessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=1, beta=1.0, kernel=rbf_kernel,\
                 kernel_gd=None, optimize=False, params_bounds=((0, 5), (1e-5, 0.1))):
        self.gamma = gamma
        self.beta = beta
        self.kernel = kernel
        self.kernel_gd = kernel_gd
        self.optimize = optimize
        self.params_bounds = params_bounds
        self._history = []

    def neg_log_likelihood(self, hyperparams, X, y):
        alpha, L, _ = self._fit(hyperparams, X, y)
        data_term = -0.5*(y.T @ alpha)
        K_term = -0.5*np.log(np.diag(L)).sum()
        const_term = -0.5*self._n_dim*np.log(2*np.pi)
        value = -(data_term + K_term + const_term)
        self._history.append(value)
        return value

    def neg_log_likelihood_prime(self, hyperparams, X, y):
        alpha, L, K_gd_on_gamma = self._fit(hyperparams, X, y)
        inv_K = cho_solve((L, True), np.eye(self._n_dim)).T
        reduce_K = np.outer(alpha, alpha) - inv_K 
        gd_on_gamma = 0.5*np.trace(reduce_K @ K_gd_on_gamma)
        gd_on_beta = 0.5*np.trace(reduce_K)
        return -np.array([gd_on_gamma, gd_on_beta])

    def _fit(self, hyperparams, X, y):
        gamma, beta = hyperparams
        K, K_gd_on_gamma = self.kernel(gamma, X, X, gradient_on_gamma=True)
        K[np.diag_indices_from(K)] += beta
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'beta' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel,) + exc.args
            raise
        coef_ = cho_solve((L, True), y)
        return coef_, L, K_gd_on_gamma

    def fit(self, X, y, verbose=False):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        if y.ndim > 1:
            y = y.ravel()
        self._n_dim = len(y)
        self._ytrain_mean = np.mean(y)
        self.X_train_, self.y_train_ = X, y-self._ytrain_mean
        hyperparams = np.array([self.gamma, self.beta])
        if self.optimize:
            res = minimize(self.neg_log_likelihood, hyperparams, args=(self.X_train_, self.y_train_),\
                           method='L-BFGS-B',\
                           jac=self.neg_log_likelihood_prime,\
                           bounds=self.params_bounds, options={'disp': verbose})
            if res.success is False:
                print('optimization %s' %(res.success))
            hyperparams = res.x
        self.gamma, self.beta = hyperparams
        self.coef_, L, _ = self._fit(hyperparams, self.X_train_, self.y_train_)
        self.inv_cov_train_ = cho_solve((L, True), np.eye(self._n_dim)).T
        return self
        
    def predict(self, X, return_variance=False):
        X = check_array(X)
        kT = self.kernel(self.gamma, X, self.X_train_)
        y_pred = self._ytrain_mean + kT @ self.coef_
        cov_pred = self.kernel(self.gamma, X, X) - kT @ self.inv_cov_train_ @ kT.T
        if return_variance:
            return y_pred, np.sqrt(np.diag(cov_pred))
        else:
            return y_pred

    def predict_gradient(self, X):
        assert self.kernel_gd is not None, print('compute gradient need kernel gradient !')
        X = check_array(X)
        n_samples, n_features = X.shape
        dy_pred = self.kernel_gd(self.gamma, X, self.X_train_) @ self.coef_
        return dy_pred.reshape((n_samples, n_features))
