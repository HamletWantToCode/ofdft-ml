# Common math functions

import numpy as np

def svd_inv(X):
    if np.all(X==0):
        X_inv = X
    else:
        U, S, Vh = np.linalg.svd(X)
        X_inv = Vh.T.conj() @ np.diag(1./S) @ U.T.conj()
    return X_inv

## used for real symmetric matrix
def svd_solver(A, b, cond=None):
    n_dims = A.shape[0]
    U, S, _ = np.linalg.svd(A)
    if cond is None:
        cond = np.finfo(np.float64).eps
    rank = len(S[S>cond])
    coefficients = np.squeeze(U.T[:rank] @ b) / S[:rank]
    x = np.sum(coefficients[np.newaxis, :] * U[:, :rank], axis=1)
    return x

# used for Hermitian positive-definite matrix
def cholesky_solver(L, b):
    n_dims = len(L)
    Lh = L.T.conj()
    y = np.zeros(n_dims)
    for i in range(n_dims):
        y[i] = (b[i] - L[i] @ y) / L[i, i]
    x = np.zeros(n_dims)
    for i in range(n_dims-1, -1, -1):
        x[i] = (y[i] - Lh[i] @ x) / Lh[i, i]
    return x

# math
def euclidean_distance(X, Y):
    # special care about complex entry
    X_ = X[:, np.newaxis, :]
    D_ = X_ - Y
    distance = np.sum(D_*D_.conj(), axis=2, dtype=np.float64)
    return np.sqrt(distance)
