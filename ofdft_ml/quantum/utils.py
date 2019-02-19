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
def kpotential_gen(nbasis, n_cos, low_V0, high_V0, low_Phi0, high_Phi0, random_state):
    """
    Generate periodic potential as sum of cosin functions, differ in their
    magnitude and phase.

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
    >>> low_phi0, high_phi0 = -0.1, 0.1
    >>> params_gen = simple_potential_gen(n_basis, low_V0, high_V0, low_phi0, high_phi0, random_state=9302)
    >>> hamilton_mat, Vq = next(params_gen)
    """
    R = np.random.RandomState(random_state)
    while True:
        Vq = np.zeros(nbasis, dtype=np.complex64)
        hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)

        V0 = R.uniform(low_V0, high_V0, n_cos)
        Phi0 = R.uniform(low_Phi0, high_Phi0, n_cos)

        Vq[0] = -np.sum(V0)
        Vq[1] = 0.5*(V0 @ np.exp(2j*np.pi*Phi0))
        np.fill_diagonal(hamilton_mat[1:, :-1], Vq[1].conj())

        yield(hamilton_mat, Vq)

# potential in k with two peaks
def special_potential_gen(nbasis, low_a, high_a, b1_range, b2_range, low_c, high_c, random_state):
    R = np.random.RandomState(random_state)
    X = np.linspace(0, 1, 100)
    while True:
        a = R.uniform(low_a, high_a, 2)
        b = np.append(R.uniform(*b1_range, 1), R.uniform(*b2_range, 1)).reshape((2, 1))
        c = R.uniform(low_c, high_c, (2, 1))
        Vx = -a @ np.exp(-(X - b)**2/(2*c**2))
        Vq = np.fft.rfft(Vx)/100
        Vq_ = Vq[:nbasis]

        hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
        for i in range(nbasis):
            np.fill_diagonal(hamilton_mat[i:, :-i], Vq_[i].conj())
        
        yield(hamilton_mat, Vq_)

def two_cosin_peak_gen(nbasis, V1_range, V2_range, V3_range, phi1_range, phi2_range, phi3_range, random_state):
    R = np.random.RandomState(random_state)
    while True:
        Vq = np.zeros(nbasis, dtype=np.complex64)
        hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)

        V0 = np.append(R.uniform(*V1_range, 1), R.uniform(*V2_range, 1), R.uniform(*V3_range, 1))
        Phi0 = np.append(R.uniform(*phi1_range, 1), R.uniform(*phi2_range, 1), R.uniform(*phi3_range, 1))

        Vq[0] = -np.sum(V0)
        Vq[1] = 0.5*(V0 @ np.exp(2j*np.pi*Phi0))
        np.fill_diagonal(hamilton_mat[1:, :-1], Vq[1].conj())

        yield(hamilton_mat, Vq)