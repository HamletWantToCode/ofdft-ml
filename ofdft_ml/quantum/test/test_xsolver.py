# test xsolver

import numpy as np 
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from ofdft_ml.quantum.solver import xsolver

n_points = 1000

def harmonicPotential(n_points, omega):
    X = np.linspace(0, 1, n_points, endpoint=True)
    # potential
    Vx = 0.5*(omega**2)*((X-0.5)**2)
    # hamiltonian
    H = np.zeros((n_points-2, n_points-2))
    np.fill_diagonal(H, -2)
    np.fill_diagonal(H[1:, :-1], 1)
    np.fill_diagonal(H[:-1, 1:], 1)
    H *= -0.5*(n_points-1)**2
    H += np.diag(Vx[1:-1])
    return H, Vx

def squarewllPotential(n_points):
    Vx = np.zeros(n_points)
    # hamiltonian
    H = np.zeros((n_points-2, n_points-2))
    np.fill_diagonal(H, -2)
    np.fill_diagonal(H[1:, :-1], 1)
    np.fill_diagonal(H[:-1, 1:], 1)
    H *= -0.5*(n_points-1)**2
    H += np.diag(Vx[1:-1])
    return H, Vx

def test_harmonic_oscillator():
    """
    1. nearby energy level difference
    2. electron density when electron number equals 1 and 2
    """
    ne1, ne2 = 1, 2
    X = np.linspace(0, 1, n_points, endpoint=True)
    omega = 100*np.pi
    H, Vx = harmonicPotential(n_points, omega)
    _, densx1, En = xsolver(n_points, H, ne1, debug=True)
    true_densx1 = np.sqrt(omega/np.pi)*np.exp(-omega*(X-0.5)**2)
    assert_almost_equal(omega, abs(En[1]-En[0]), 5)
    assert_array_almost_equal(true_densx1, densx1, 5)
    _, densx2 = xsolver(n_points, H, ne2)
    true_densx_2nd = 2*np.sqrt(omega/np.pi)*omega*((X-0.5)**2)*np.exp(-omega*(X-0.5)**2)
    true_densx2 = true_densx1 + true_densx_2nd
    assert_array_almost_equal(true_densx_2nd, densx2, 5)

def test_square_well():
    """
    1. electron density
    2. kinetic energy
    """
    ne = 1
    X = np.linspace(0, 1, n_points, endpoint=True)
    H, Vx = squarewllPotential(n_points)
    Ek, densx = xsolver(n_points, H, ne)
    true_Ek = np.pi**2/2.0
    true_densx = 2*(np.sin(np.pi*X))**2
    assert_almost_equal(true_Ek, Ek, 3)
    assert_array_almost_equal(true_densx, densx, 5)