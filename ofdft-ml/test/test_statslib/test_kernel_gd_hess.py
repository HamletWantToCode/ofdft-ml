# test kernel gd

import numpy as np 
from statslib.utils import rbf_kernel, rbf_kernel_gradient, rbf_kernel_hessan

R = np.random.RandomState(43492)

train_X = R.rand(10, 3)
test_X = R.rand(5, 3)

gamma = 0.1

# verify kernel gradient
f0 = rbf_kernel(gamma, test_X, train_X)
step = np.ones(3)*1e-3
test_X_new = test_X + step
f1 = rbf_kernel(gamma, test_X_new, train_X)
gd = rbf_kernel_gradient(gamma, test_X, train_X)
f_gd = f0 +np.sum(gd*step[np.newaxis, :, np.newaxis], axis=1)
err_gd = abs(f_gd - f1)
print(err_gd)

# verify kernel hessan
f_gd_0 = rbf_kernel_gradient(gamma, test_X, train_X)
step = np.ones(3)*1e-3
train_X_new = train_X + step
f_gd_1 = rbf_kernel_gradient(gamma, test_X, train_X_new)
hess = rbf_kernel_hessan(gamma, test_X, train_X).reshape((5, 3, 10, 3))
f_hess = f_gd_0 + hess @ step
err_hess = abs(f_hess - f_gd_1)
print(err_hess)
