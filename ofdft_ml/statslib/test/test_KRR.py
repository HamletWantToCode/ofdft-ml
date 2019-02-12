import numpy as np 
from scipy.optimize import approx_fprime
from numpy.testing import assert_array_almost_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels

from ofdft_ml.statslib.kernel_ridge import KernelRidge as myKernelRidge
from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_gradient

X, y = make_regression(n_features=10, random_state=0)
n_samples, D = X.shape


def test_kernel_matrix():
    gamma = 0.1
    my = rbf_kernel(gamma, X, X)
    sklearn = pairwise_kernels(X, X, metric='rbf', gamma=gamma)
    assert_array_almost_equal(sklearn, my, 7)

def test_kernel_gradient():
    """
    test against f'(x)=(f(x+h) - f(x-h)) / 2*h + O(h^2)
    """
    gamma = 0.1
    x, x_ref = X[0], X[1]
    my = rbf_kernel_gradient(gamma, x[np.newaxis, :], x_ref[np.newaxis, :])[0, :, 0]
    numerical = []
    for i in range(D):
        h = np.zeros_like(x)
        h[i] += 1e-3
        x_f, x_b = x+h, x-h
        partial_gd = (rbf_kernel(gamma, x_f[np.newaxis, :], x_ref[np.newaxis, :]) - \
                      rbf_kernel(gamma, x_b[np.newaxis, :], x_ref[np.newaxis, :]))[0, 0] / 2e-3
        numerical.append(partial_gd)
    assert_array_almost_equal(np.array(numerical), my, 6)
    
def test_kernel_ridge():
    """
    Compared with scikit-learn "kernel_ridge" 
    """
    C, gamma = 1e-3, 0.1
    alpha = n_samples*C
    sklearn = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma).fit(X, y).predict(X)
    my = myKernelRidge(C=C, gamma=gamma, kernel=rbf_kernel).fit(X, y).predict(X)
    assert_array_almost_equal(sklearn, my, 7)

def test_predict_gradient():
    C, gamma = 1e-3, 0.1
    alpha = n_samples*C
    x = X[0]
    my = myKernelRidge(C=C, gamma=gamma, kernel=rbf_kernel,\
                       kernel_gd=rbf_kernel_gradient).fit(X, y).predict_gradient(x[np.newaxis, :])[0]
    sklearn = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    sklearn.fit(X, y)
    numeric = []
    for i in range(D):
        h = np.zeros_like(x)
        h[i] += 1e-3
        x_f, x_b = x+h, x-h
        partial_gd = (sklearn.predict(x_f[np.newaxis, :])\
                      - sklearn.predict(x_b[np.newaxis, :]))[0] / 2e-3
        numeric.append(partial_gd)
    assert_array_almost_equal(my, np.array(numeric), 4)