import numpy as np
from scipy.optimize import approx_fprime
from numpy.testing import assert_array_almost_equal
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from ofdft_ml.statslib.GaussProcess import GaussProcessRegressor as myGaussRegressor
from ofdft_ml.statslib.kernel import rbf_kernel, rbf_kernel_gradient

X, y = make_regression(n_features=10, random_state=0)
n_samples, D = X.shape

def test_gauss_process():
    sklearn_gp_rbf = RBF(length_scale=0.5, length_scale_bounds=None)
    sklearn = GaussianProcessRegressor(kernel=sklearn_gp_rbf, alpha=1e-10, optimizer=None).fit(X, y).predict(X)
    my = myGaussRegressor(gamma=2, beta=1e-10, kernel=rbf_kernel).fit(X, y).predict(X)
    assert_array_almost_equal(sklearn, my, 7)

def test_gauss_process_gradient():
    x = X[0]
    sklearn_gp_rbf = RBF(length_scale=0.5, length_scale_bounds=None)
    sklearn = GaussianProcessRegressor(kernel=sklearn_gp_rbf, alpha=1e-10, optimizer=None).fit(X, y)
    my = myGaussRegressor(gamma=2, beta=1e-10, kernel=rbf_kernel,\
                          kernel_gd=rbf_kernel_gradient).fit(X, y).predict_gradient(x[np.newaxis, :])[0]
    numeric = []
    for i in range(D):
        h = np.zeros_like(x)
        h[i] += 1e-3
        x_f, x_b = x+h, x-h
        partial_gd = (sklearn.predict(x_f[np.newaxis, :])\
                      - sklearn.predict(x_b[np.newaxis, :]))[0] / 2e-3
        numeric.append(partial_gd)
    assert_array_almost_equal(my, np.array(numeric), 4)

