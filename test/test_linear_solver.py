# test solver performance in solving ill-conditioned
# linear equations

import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as plt

from ext_math import svd_solver, cholesky_solver

R = np.random.RandomState(584392)
A = hilbert(20)
x_true = R.uniform(-10, 10, 20)
b = A @ x_true
k = 0

x_svd = svd_solver(A, b, k)
x_ch = cholesky_solver(A, b, k)

plt.plot(np.arange(0, 20, 1), x_true, 'ro--', label='true solution')
plt.plot(np.arange(0, 20, 1), x_svd, 'bo--', fillstyle='none', label='SVD solution')
plt.plot(np.arange(0, 20, 1), x_ch, 'gx--', fillstyle='none', label='Cholesky solution')
plt.legend()
plt.xlabel('#')
plt.ylabel('value')
plt.show()
