# quantum solver

import numpy as np
from scipy.linalg import eigh

def solver(nk, nbasis, hamiton_mat, occ=1, debug=False):
    """
    fix electron number equals 1
    """
    kpoints = np.linspace(0, np.pi, nk)
    # build and solve eigenvalue problem
    T = 0
    mu = 0
    if debug:
        En = np.zeros((nk, nbasis))
    density = np.zeros(nbasis, dtype=np.complex64)
    for ki, k in enumerate(kpoints):
        kinetic_term = np.array([0.5*(k+(i-nbasis//2)*2*np.pi)**2 for i in range(nbasis)])
        np.fill_diagonal(hamiton_mat, kinetic_term)
        En_k, Uq_k = eigh(hamiton_mat, overwrite_a=True, overwrite_b=True)
        if debug:
            En[ki] = En_k
        # compute mu
        if ki == 0:
            bottom = En_k[0]         # set the minimum of band energy to 0 !
        if ki == (nk-1):
            top = En_k[0]
        # compute electron density
        # compute kinetic energy
        num_mat_eigspace = np.zeros((nbasis, nbasis))
        num_mat_eigspace[0, 0] = 1
        # for i in range(nbasis):
        #     if (i+1) <= occ*nk:
        #         num_mat_eigspace[i, i] = 1
        #     else:
        #         mu = En_k[i-1] - b
        #         break
        density_mat_kspace = Uq_k @ (num_mat_eigspace @ (Uq_k.T).conj())

        density_k = np.zeros(nbasis, dtype=np.complex64)
        T_k = 0
        for i in range(nbasis):
            density_k[i] = np.trace(density_mat_kspace, offset=i)
            T_k += 0.5*((k+(i-nbasis//2)*2*np.pi)**2)*(density_mat_kspace[i, i]).real
        T += T_k
        density += density_k
    mu = top - bottom
    if debug:
        return T/nk, density/nk, mu, En
    else:
        return T/nk, density/nk, mu




