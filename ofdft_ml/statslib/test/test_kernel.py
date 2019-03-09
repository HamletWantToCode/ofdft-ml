import numpy as np 
from scipy.optimize import approx_fprime
from numpy.testing import assert_array_almost_equal
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_regression

from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_gradient

X, y = make_regression(n_features=10, random_state=0)
n_samples, D = X.shape

def test_kernel_matrix():
    gamma = 0.02
    my = rbf_kernel(gamma, X, X)
    sklearn_common = pairwise_kernels(X, X, metric='rbf', gamma=gamma)
    sklearn_gp = RBF(length_scale=5, length_scale_bounds=None).__call__(X, X)
    assert_array_almost_equal(sklearn_common, my, 7)
    assert_array_almost_equal(sklearn_gp, my, 7)

def test_kernel_gradient_on_hyper_params():
    """
    f'(x)=(f(x+h)-f(x)) / h + O(h)
    """
    gamma = 5
    K, my = rbf_kernel(gamma, X, X, gradient_on_gamma=True)
    gamma_ = gamma + 1e-3
    K_ = rbf_kernel(gamma_, X, X)
    numeric = (K_ - K) / 1e-3
    assert_array_almost_equal(numeric, my, 3)

def test_kernel_gradient():
    """
    test against f'(x)=(f(x+h) - f(x-h)) / 2*h + O(h^2)
    """
    gamma = 0.1
    x, x_ref = X[0], X[1]
    my = rbf_kernel_gradient(gamma, x[np.newaxis, :], x_ref[np.newaxis, :])[:, 0]
    numerical = []
    for i in range(D):
        h = np.zeros_like(x)
        h[i] += 1e-3
        x_f, x_b = x+h, x-h
        partial_gd = (rbf_kernel(gamma, x_f[np.newaxis, :], x_ref[np.newaxis, :]) - \
                      rbf_kernel(gamma, x_b[np.newaxis, :], x_ref[np.newaxis, :]))[0, 0] / 2e-3
        numerical.append(partial_gd)
    assert_array_almost_equal(np.array(numerical), my, 6)