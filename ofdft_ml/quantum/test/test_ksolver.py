# test ksolver

import numpy as np
from numpy.testing import assert_almost_equal

from ofdft_ml.quantum.solver import ksolver

nk = 100
nbasis = 10
occ = 1

def kpotential(nbasis, V0):
    """
    V(x) = V_0(\cos^2(\pi x)-1) (V_0>=0)
    """
    hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
    Vq = np.zeros(nbasis, dtype=np.complex64)
    Vq[0], Vq[1] = -0.5*V0, 0.25*V0
    np.fill_diagonal(hamilton_mat[1:, :-1], Vq[1])
    return hamilton_mat, Vq

def test_band_energy():
    """
    Near free electron approximation, gap=2|V(q=1)|
    """
    V0 = 0.1
    H, Vq = kpotential(nbasis, V0)
    _, _, _, En = ksolver(nk, nbasis, H, occ, debug=True)
    gap = En[-1, 1] - En[-1, 0]
    assert_almost_equal(2*abs(Vq[1]), gap, 5)

def test_kinetic_energy():
    """
    Thomas-Fermi approximation in 1D
    """
    V0 = 0.0
    nk = 10000
    hamilton_mat, Vq = kpotential(nbasis, V0)
    Ek_compute, densq, mu = ksolver(nk, nbasis, hamilton_mat, occ)
    # check electron number
    assert_almost_equal(occ, densq[0])
    # check Ek
    Ek_TF = (np.pi**2/6)*(occ**3)
    assert_almost_equal(Ek_TF, Ek_compute, 2)
    # assert abs(Ek_compute-Ek_TF) < 10.0/nk, print('error in kinetic energy computation')

def test_electron_density():
    pass
