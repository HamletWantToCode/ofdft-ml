# miscellanious utility

import numpy as np 
from ..ext_math import euclidean_distance

# kernel
def rbf_kernel(gamma, X, Y, gradient_on_gamma=False):
    square_distance = (euclidean_distance(X, Y))**2
    K = np.exp(-gamma*square_distance)
    if gradient_on_gamma:
        K_gd_gamma = -square_distance * K
        return K, K_gd_gamma
    else:
        return K

def rbf_kernel_gradient(gamma, X, Y):
    n_samples_X, n_features_X = X.shape
    n_samples_Y, n_features_Y = Y.shape
    assert n_features_X == n_features_Y, print('feature dimension of train and predict data mismatch')
    square_distance = (euclidean_distance(X, Y))**2
    K = np.exp(-gamma*square_distance)
    # column_matrix_list = []
    # for i in range(n_samples_X):
    #     row_matrix_list = []
    #     for j in range(n_samples_Y):
    #         x_diff = X[i] - Y[j]
    #         inner_vector = -2*gamma*x_diff*K[i, j]
    #         row_matrix_list.append(inner_vector)
    #     column_matrix_list.append(row_matrix_list)
    # K_gd = np.block(column_matrix_list).T
    diff = X[:, :, np.newaxis] - Y.T 
    K_gd = -2*gamma*diff*K[:, np.newaxis, :]
    return K_gd.reshape((-1, n_samples_Y))

def rbf_kernel_2nd_gradient(gamma, X, Y, gradient_on_gamma=False):
    """
    "gradient_on_gamma" keyword is added for compatibility
    """
    n_samples_X, n_features_X = X.shape
    n_samples_Y, n_features_Y = Y.shape
    assert n_features_X == n_features_Y, print('feature dimension of train and predict data mismatch')
    square_distance = (euclidean_distance(X, Y))**2
    K = np.exp(-gamma*square_distance)
    column_matrix_list = []
    for i in range(n_samples_X):
        row_matrix_list = []
        for j in range(n_samples_Y):
            x_diff = X[i] - Y[j]
            inner_matrix = 2*gamma*K[i, j]*(np.eye(n_features_X) - 2*gamma*np.outer(x_diff, x_diff))
            row_matrix_list.append(inner_matrix)
        column_matrix_list.append(row_matrix_list)
    K_2nd_gradient = np.block(column_matrix_list)
    if gradient_on_gamma is True:
        return K_2nd_gradient, None
    else:
        return K_2nd_gradient

def approximate_rbf_kernel_hessian(gamma, X, Y, gradient_on_gamma=False):
    K = rbf_kernel(gamma, X, Y, gradient_on_gamma=False)
    if gradient_on_gamma:
        return gamma*K, None
    else:
        return gamma*K