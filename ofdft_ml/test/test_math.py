# test linear equation solver & Euclidean distance

import numpy as np
from numpy.testing import assert_array_almost_equal

from ofdft_ml.ext_math import euclidean_distance

# def test_normal_matrix():
#     # matrix should be positive definite symmetric metrix !!!
#     A = np.array([[2.0, -1.0],
#                   [-1.0, 0.8]])
#     true_x = np.array([-0.3, 0.07])
#     b = A @ true_x
#     x_svd = svd_solver(A, b)
#     x_ch = cholesky_solver(A, b)
#     assert_array_almost_equal(x_svd, x_ch, 7)
#     assert_array_almost_equal(true_x, x_svd, 7)
#
# def test_ill_condition_matrix():
#     from scipy.linalg import hilbert
#     A = hilbert(10)
#     true_x = np.array([ 8.47645604,  6.90407127,  7.53306828,  0.8399532 , -2.82329886,\
#                      3.4190595 , -5.50868032,  5.48982295, -0.75044064,  5.14573201])
#     b = A @ true_x
#     x_svd = svd_solver(A, b)
#     x_ch = cholesky_solver(A, b)
#     assert_array_almost_equal(x_svd, x_ch, 5)
#     assert_array_almost_equal(true_x, x_svd)

def test_Euclidean_distance():
    R = np.random.RandomState(0)
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    true_eu_dist_mat = np.array([[0, 3*np.sqrt(3)],
                                 [3*np.sqrt(3), 0]])
    compute_eu_dist = euclidean_distance(X, X)
    assert_array_almost_equal(true_eu_dist_mat, compute_eu_dist, 7)
