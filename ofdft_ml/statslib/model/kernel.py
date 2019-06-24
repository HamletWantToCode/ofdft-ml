import numpy as np 

class RBFGrad_Kernel(object):
    def __init__(self, gamma, n_tasks):
        self.gamma = gamma
        self.n_tasks = n_tasks
        self.kernel_func = RBF_Kernel(gamma)
        
    def index_kernel(self, X, Y):
        n_X, n_Y = len(X), len(Y)
        K = np.zeros((n_X, n_Y, self.n_tasks))
        for i in range(self.n_tasks):
            dxx1 = X[:, i][:, np.newaxis] - Y[:, i][np.newaxis, :]
            K[:, :, i] = 2*self.gamma*(1 - 2*self.gamma*dxx1**2)
        return K
    
    def __call__(self, X, Y):
        k_xy = self.kernel_func(X, Y)
        K_ij = self.index_kernel(X, Y)
        return K_ij * k_xy[:, :, np.newaxis]

class RBF_Kernel(object):
    def __init__(self, gamma):
        self._gamma = gamma

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value > 0:
            self._gamma = value
        else:
            raise ValueError('Gamma must larger than 0 !')

    def __call__(self, X, Y, eval_gradient=False):
        square_distance = (euclidean_distance(X, Y))**2
        K = np.exp(-self.gamma*square_distance)
        if eval_gradient:
            K_gd_gamma = -square_distance * K
            return K, K_gd_gamma
        else:
            return K

def euclidean_distance(X, Y):
    # special care about complex entry
    X_ = X[:, np.newaxis, :]
    D_ = X_ - Y
    distance = np.sum(D_*D_.conj(), axis=2, dtype=np.float64)
    return np.sqrt(distance)

class Multitask_Kernel(object):
    def __init__(self, gamma, n_tasks):
        self.gamma = gamma
        self.n_tasks = n_tasks
        self.kernel_func = RBF_Kernel(gamma)
        
    def index_kernel(self, X, Y):
        n_X, n_Y = len(X), len(Y)
        K = np.zeros((n_X, n_Y, self.n_tasks))
        for i in range(self.n_tasks):
            K[:, :, i] = np.ones((n_X, n_Y))
        return K
    
    def __call__(self, X, Y):
        k_xy = self.kernel_func(X, Y)
        K_ij = self.index_kernel(X, Y)
        return K_ij * k_xy[:, :, np.newaxis]



def rbf_kernel_gradient(gamma, X, Y):
    n_samples_X, n_features_X = X.shape
    n_samples_Y, n_features_Y = Y.shape
    assert n_features_X == n_features_Y, print('feature dimension of train and predict data mismatch')
    square_distance = (euclidean_distance(X, Y))**2
    K = np.exp(-gamma*square_distance)
    diff = X[:, :, np.newaxis] - Y.T 
    K_gd = -2*gamma*diff*K[:, np.newaxis, :]
    return K_gd.reshape((-1, n_samples_Y))

# def rbf_kernel(gamma, X, Y, eval_gradient=False):
#     square_distance = (euclidean_distance(X, Y))**2
#     K = np.exp(-gamma*square_distance)
#     if eval_gradient:
#         K_gd_gamma = -square_distance * K
#         return K, K_gd_gamma
#     else:
#         return K

# def partial_derivative_gp_rbf_kernel(gamma, X, Y, gradient_on_gamma=False, index=None):
#     n_samples_X, n_samples_Y = X.shape[0], Y.shape[0]
#     K = rbf_kernel(gamma, X, Y, gradient_on_gamma=False)
#     X_i, Y_i = X[:, index], Y[:, index]
#     prefactor = np.ones((n_samples_X, n_samples_Y)) - 2*gamma*(X_i[:, np.newaxis]-Y_i[np.newaxis, :])**2
#     hessian = 2*gamma*prefactor*K
#     if gradient_on_gamma:
#         return hessian, None
#     else:
#         return hessian

# def rbf_kernel_2nd_gradient(gamma, X, Y, gradient_on_gamma=False, index=None):
#     """
#     "gradient_on_gamma" keyword is added for compatibility
#     """
#     n_samples_X, n_features_X = X.shape
#     n_samples_Y, n_features_Y = Y.shape
#     assert n_features_X == n_features_Y, print('feature dimension of train and predict data mismatch')
#     square_distance = (euclidean_distance(X, Y))**2
#     K = np.exp(-gamma*square_distance)
#     column_matrix_list = []
#     for i in range(n_samples_X):
#         row_matrix_list = []
#         for j in range(n_samples_Y):
#             x_diff = X[i] - Y[j]
#             inner_matrix = 2*gamma*K[i, j]*(np.eye(n_features_X) - 2*gamma*np.outer(x_diff, x_diff))
#             row_matrix_list.append(inner_matrix)
#         column_matrix_list.append(row_matrix_list)
#     K_2nd_gradient = np.block(column_matrix_list)
#     if gradient_on_gamma is True:
#         return K_2nd_gradient, None
#     else:
#         return K_2nd_gradient