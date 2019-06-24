from ofdft_ml.statslib.base import BaseGP
from .kernel import RBF_Kernel, RBFGrad_Kernel # Multitask_Kernel
import numpy as np 
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
import warnings

class ScalarGP(BaseGP):
    def __init__(self, gamma, noise, bounds):
        super(ScalarGP, self).__init__(noise, RBF_Kernel, None)
        self.gamma = gamma
        self.kernel_func = self.kernel(gamma)
        self.bounds = bounds
        self._history = []

    def neg_log_likelihood(self, hyperparameters, train_x, train_y):
        gamma, noise = hyperparameters
        self.kernel_func.gamma = gamma
        K = self.kernel_func(train_x, train_x)
        K += noise*np.eye(K.shape[0])
        L = np.linalg.cholesky(K)
        coef = cho_solve((L, True), train_y)

        data_term = -0.5*(train_y.T @ coef)
        L_diag = np.einsum('ii->i', L)
        K_term = -(np.log(L_diag)).sum()
        const_term = -0.5*(np.log(2*np.pi))*len(train_x)
        value = -(data_term + K_term + const_term)
        self._history.append(value)
        return value

    def neg_log_likelihood_gradient(self, hyperparameters, train_x, train_y):
        gamma, noise = hyperparameters
        self.kernel_func.gamma = gamma
        K, K_gradient = self.kernel_func(train_x, train_x, eval_gradient=True)
        K += noise*np.eye(K.shape[0])
        L = np.linalg.cholesky(K)
        coef = cho_solve((L, True), train_y)
        inv_L = solve_triangular(L, np.eye(K.shape[0]), lower=True)
        inv_K = inv_L.T @ inv_L

        reduce_K = np.outer(coef, coef) - inv_K
        gd_on_gamma = 0.5*np.trace(reduce_K @ K_gradient)
        gd_on_noise = 0.5*np.trace(reduce_K)
        return -np.array([gd_on_gamma, gd_on_noise])

    def fit(self, train_x, train_y, verbose=False):
        if self.mean_f is not None:
            self._mean = self.mean_f(train_x)
        else:
            warnings.warn("use constant mean function")
            self._mean = np.mean(train_y)
        train_y_ = train_y - self._mean

        res = minimize(self.neg_log_likelihood,
                       np.array([self.gamma, self.noise]),
                       args=(train_x, train_y),
                       method='L-BFGS-B',
                       jac=self.neg_log_likelihood_gradient,
                       bounds=self.bounds,
                       options={'disp': verbose,
                                'ftol': 1e-5,
                                'maxiter': 1000})

        if res.success:
            hyperparameters = res.x
        else:
            warnings.warn('BFGS not converge successfully, use initial hyperparameter values !')
            hyperparameters = np.array([self.gamma, self.noise])
        self.gamma, self.noise = hyperparameters
        self.kernel_func.gamma = self.gamma
        K = self.kernel_func(train_x, train_x)
        L = np.linalg.cholesky(K + self.noise*np.eye(K.shape[0]))
        
        self.coef = cho_solve((L, True), train_y)
        self._L = L
        self._train_x = train_x

    def predict(self, x, return_std=False):
        k_trans = self.kernel_func(x, self._train_x)
        y = self._mean + k_trans @ self.coef
        if return_std:
            inv_L = solve_triangular(self._L, np.eye(self._L.shape[0]), lower=True)
            inv_K = inv_L.T @ inv_L
            y_std = np.sqrt(np.diag(self.kernel_func(x, x)) - np.einsum('ij,ij->i', k_trans @ inv_K, k_trans))
            return y, y_std
        else:
            return y


class MultitaskGP(BaseGP):
    def __init__(self, gamma, noise, n_tasks):
        super(MultitaskGP, self).__init__(noise, RBFGrad_Kernel, None)
        self.gamma = gamma
        self.kernel_func = RBFGrad_Kernel(gamma, n_tasks)
        # self.kernel_func = Multitask_Kernel(gamma, n_tasks)
        self.n_tasks = n_tasks

    def fit(self, train_x, train_y):
        if self.mean_f is not None:
            self._mean = self.mean_f(train_x)
        else:
            warnings.warn("use constant mean function")
            self._mean = np.mean(train_y, axis=0)
        train_y_ = train_y - self._mean

        Ks = self.kernel_func(train_x, train_x)
        coefs = np.zeros((Ks.shape[0], self.n_tasks))
        for i in range(self.n_tasks):
            L = np.linalg.cholesky(Ks[:, :, i] + self.noise*np.eye(Ks.shape[0]))
            coef = cho_solve((L, True), train_y[:, i])
            coefs[:, i] = coef

        self.coefs = coefs
        self._train_x = train_x

    def predict(self, x):
        ks_trans = self.kernel_func(x, self._train_x)
        y = self._mean + np.einsum('ijk,jk->ik', ks_trans, self.coefs)
        return y