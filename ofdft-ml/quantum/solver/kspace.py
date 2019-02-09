# quantum solver

import numpy as np
from scipy.linalg import eigh

def ksolver(nk, nbasis, hamiton_mat, occ=1, debug=False):
    """
    Solving for groundstate electron density & kinetic energy given a hamiltonian matrix
    writen in k-space.

    :param nk: number of k points
    :param nbasis: number of plane wave basis, dimension of hamilton matrix
    :param occ: occupation number, integer
    :param debug: default False, swich on to record the energy band

    :return T: a tuple contains electron kinetic energy per cell & groundstate electron density in k space
               & chemical potential

    Examples::

    >>> from quantum.solver import solver
    >>> import numpy as np
    >>> n_kpoints = 100
    >>> n_basis = 10
    >>> sample_hamilton_mat = np.zeros((n_basis, n_basis), dtype=np.complex64)
    >>> np.fill_diagonal(sample_hamilton_mat[1:, :-1], -5+0j)
    >>> T, densq, mu = solver(n_kpoints, n_basis, sample_hamilton_mat)
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
            top = En_k[0]
        # compute electron density
        # compute kinetic energy
        num_mat_eigspace = np.zeros((nbasis, nbasis))
        for i in range(occ):
            num_mat_eigspace[i, i] = 1
            if En_k[i] > top:
                top = En_k[i]

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




