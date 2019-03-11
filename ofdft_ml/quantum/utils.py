import numpy as np

# finite range potential in x space
def xpotential_gen(n_points, n_Gauss, low_a, high_a, low_b, high_b, low_c, high_c, random_state):
    R = np.random.RandomState(random_state)
    X = np.linspace(0, 1, n_points, endpoint=True)
    while True:
        # potential
        a = R.uniform(low_a, high_a, n_Gauss)
        b = R.uniform(low_b, high_b, (n_Gauss, 1))
        c = R.uniform(low_c, high_c, (n_Gauss, 1))
        Vx = -a @ np.exp(-(X - b)**2/(2*c**2))
        # hamiltonian matrix
        H = np.zeros((n_points-2, n_points-2))
        np.fill_diagonal(H, -2)
        np.fill_diagonal(H[1:, :-1], 1)
        np.fill_diagonal(H[:-1, 1:], 1)
        H *= -0.5*(n_points-1)**2
        H += np.diag(Vx[1:-1])
        yield(H, Vx)

# periodic potential in k space
def kpotential_gen(nbasis, n_gauss, low_V0, high_V0, low_mu, high_mu, low_l, high_l, random_state):
    """
    Generate periodic potential as sum of gauss-like functions, differ in their
    magnitude, mean & length scale.

    :param nbasis: number of plane wave basis, related to the size of hamilton matrix
    :param n_cos: number of cosin function used
    :param low_V0: lower bound for cosin function's magnitude
    :param high_V0: upper bound for cosin function's magnitude
    :param random_state: an integer as random seed

    :return: Generator generates hamitonian matrix and potential k-components in each iteration

    Examples::

    >>> from quantum.utils import simple_potential_gen
    >>> n_basis = 10
    >>> low_V0, high_V0 = 3, 5
    >>> low_mu, high_mu = -0.1, 0.1
    >>> low_l, high_l = 0.03, 0.1
    >>> params_gen = kpotential_gen(n_basis, low_V0, high_V0, low_phi0, high_phi0, random_state=9302)
    >>> hamilton_mat, Vq = next(params_gen)
    """
    R = np.random.RandomState(random_state)
    while True:
        V0 = R.uniform(low_V0, high_V0, n_gauss)
        Mu = R.uniform(low_mu, high_mu, n_gauss)
        L = R.uniform(low_l, high_l, n_gauss)
        Vq = np.zeros(nbasis, dtype=np.complex64)
        k_grids = np.arange(0, nbasis, 1)
        for i in range(n_gauss):
            Vq += -np.sqrt(2*np.pi)*V0[i]*L[i]*np.exp(-2*(np.pi*L[i]*k_grids)**2)*np.exp(-2j*np.pi*Mu[i]*k_grids)

        hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
        for i in range(nbasis):
            np.fill_diagonal(hamilton_mat[i:, :-i], Vq[i].conj())

        yield(hamilton_mat, Vq)