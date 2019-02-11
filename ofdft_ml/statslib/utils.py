# miscellanious utility

import numpy as np 
from ..ext_math import euclidean_distance

# kernel
def rbf_kernel(gamma, X, Y):
    square_distance = (euclidean_distance(X, Y))**2
    return np.exp(-gamma*square_distance)

def rbf_kernel_gradient(gamma, X, Y):
    square_distance = (euclidean_distance(X, Y))**2
    K = np.exp(-gamma*square_distance)
    diff = X[:, :, np.newaxis] - Y.T 
    K_gd = -2*gamma*diff*K[:, np.newaxis, :]
    return K_gd

def rbf_kernel_hessian(gamma, X, Y):
    _, n_dims = X.shape
    square_distance = (euclidean_distance(X, Y))**2
    K = np.exp(-gamma*square_distance)
    diff = X[:, :, np.newaxis] - Y.T 
    K_hessian = 2*gamma*(np.eye(n_dims)[np.newaxis, :, :, np.newaxis] - \
                         2*gamma*diff[:, :, np.newaxis, :]*diff[:, np.newaxis, :, :])\
                *K[:, np.newaxis, np.newaxis, :]
    return K_hessian

# def rbf_kernel_hessan(gamma, X, Y):
#     (N_ts, D), N_tr = X.shape, Y.shape[0]
#     square_distance = (euclidean_distance(X, Y))**2
#     K = np.exp(-gamma*square_distance)
#     K_hess = np.zeros((N_ts*D, N_tr*D))
#     E = np.eye(D)
#     for i in range(0, N_ts*D, D):
#         m = i//D
#         for j in range(0, N_tr*D, D):
#             n = j//D
#             diff = X[m] - Y[n] 
#             K_hess[i:i+D, j:j+D] = (E - 2*gamma*diff[:, np.newaxis]*diff[np.newaxis, :])*K[m, n]
#     K_hess *= 2*gamma
#     return K_hess