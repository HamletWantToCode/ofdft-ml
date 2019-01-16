# test hamiltonian

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import eig_banded

Vq = np.array([-0.5, -0.25])*10
nbasis = 20
max_pos_freq = len(Vq)
hamiton_mat = np.zeros((max_pos_freq, nbasis), dtype=np.complex64)
for i in range(max_pos_freq):
    for j in range(i, nbasis):
        if i==0:
            hamiton_mat[max_pos_freq-1, j] = Vq[0]
        else:
            hamiton_mat[max_pos_freq-i-1, j] = Vq[i]

h = np.zeros((nbasis, nbasis))
for i in range(nbasis):
    h[i, i] = Vq[0]
    for j in range(i+1, nbasis):
        if j-i>1:
            break
        else:
            h[i, j] = h[j, i] = Vq[1]

En_k, Uq_k = eig_banded(hamiton_mat, overwrite_a_band=True, select='a')
w, v = np.linalg.eigh(h)

print(w - En_k)
plt.matshow(v.real - Uq_k.real)
plt.colorbar()
plt.show()
