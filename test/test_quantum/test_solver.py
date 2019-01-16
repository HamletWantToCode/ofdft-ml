# test solver

import numpy as np
import matplotlib.pyplot as plt 

from quantum.solver import solver

nk = 100
nbasis = 10

def params_gen(nbasis, V0):
    hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
    Vq = np.zeros(nbasis, dtype=np.complex64)
    Vq[0], Vq[1] = -0.5*V0, -0.25*V0
    np.fill_diagonal(hamilton_mat[1:, :-1], Vq[1])
    return hamilton_mat, Vq

## Thomas-Fermi test
occ = 1
hamilton_mat, Vq = params_gen(nbasis, 0)
Ek_compute, densq, mu, En = solver(nk, nbasis, hamilton_mat, occ, debug=True)
assert densq[0].real == 1, print('electron number incorret')
Ek_TF = (np.pi**2/6)*(occ**3)
assert abs(Ek_compute-Ek_TF) < 10.0/nk, print('error in kinetic energy computation')
fig1 = plt.figure()
ax1 = fig1.gca()
k_points = np.linspace(0, np.pi, nk)
for i in range(3):
    ax1.plot(k_points, En.T[i], 'b')
ax1.axhline(mu, 0, np.pi, color='r', linestyle='--')
ax1.set_xlabel('k')
ax1.set_ylabel('E(k)')
plt.savefig('test/test_quantum/free_e_band.png')

## near free electron band gap test
fig2 = plt.figure()
ax2 = fig2.gca()
BAND_GAP = []
Vk = []
for V0 in np.logspace(-3, 3, 10):
    hamilton_mat, Vq = params_gen(nbasis, V0)
    _, _, _, En = solver(nk, nbasis, hamilton_mat, debug=True) 
    BAND_GAP.append(En[-1, 1] - En[-1, 0])
    Vk.append(2*abs(Vq[1]))
ax2.plot(Vk, BAND_GAP, 'ro')
ax2.plot(Vk, Vk, 'b')
ax2.semilogx()
ax2.semilogy()
ax2.set_xlabel(r'$2|\hat{V}(1)|$')
ax2.set_ylabel(r'$\Delta E(1)$')
plt.savefig('test/test_quantum/near_free_band_gap.png')

## harmonic oscillator limit test
hamilton_mat, Vq = params_gen(nbasis, 100)
_, densq, _, En = solver(nk, nbasis, hamilton_mat, debug=True)
assert densq[0].real == 1, print('electron number incorrect')
omega = 2*np.pi*np.sqrt(100)
true_dens = lambda x: np.sqrt(omega/np.pi)*np.exp(-omega*x**2)
X = np.linspace(0, 1, 100)
y = true_dens(X-0.5)

fig3 = plt.figure()
ax3 = fig3.gca()
densx = np.fft.irfft(densq, 100)*100
densx = np.fft.fftshift(densx)
ax3.plot(X, y, 'r', label='true')
ax3.plot(X, densx, 'b', label='compute')
plt.savefig('test/test_quantum/harmonic_atom_limit.png')