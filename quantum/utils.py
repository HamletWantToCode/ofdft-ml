import numpy as np

# simulate periodic potential
def simple_potential_gen(nbasis, n_cos, low_V0, high_V0, low_Phi0, high_Phi0, random_state):
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

# potential generator
# def potential_gen(nbasis, max_q, low_V0, high_V0, low_mu, high_mu, random_state):
#     np.random.seed(random_state)
#     assert max_q > 2
#     NG = np.arange(2, max_q, 1, 'int')
#     while True:
#         nq = np.random.randint(0, max_q-2)      # nq is number of non-zero k components other than 0 and 1 component
#         if nq == 0:
#             q_index = np.array([1])
#         else:
#             q_index = np.append(np.random.choice(NG, size=nq), 1)
#         mu = np.random.uniform(low_mu, high_mu)
#         Vq = np.zeros(nbasis, dtype=np.complex64)
#         hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
#         V0 = np.random.uniform(low_V0, high_V0)
#         for i in q_index:
#             theta = np.random.uniform(0, 2*np.pi)
#             r0 = np.random.rand()
#             Vq_conj = -V0*r0*(np.cos(theta) - 1j*np.sin(theta))
#             Vq[i] = Vq_conj.conjugate()
#             np.fill_diagonal(hamilton_mat[i:, :-i], Vq_conj)
#         yield (hamilton_mat, Vq, mu)

