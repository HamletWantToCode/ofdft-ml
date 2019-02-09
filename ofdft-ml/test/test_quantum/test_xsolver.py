import numpy as np 
from quantum.solver import xsolver
import matplotlib.pyplot as plt 

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

N_points = 100
N_electron = 2

# harmonic oscillator
omega = 100*np.pi  
H_har, Vx_har = harmonicPotential(N_points, omega)
Tk_har, densx_har = xsolver(N_points, H_har, N_electron)

# kinetic energy
H_sqr, Vx_sqr = squarewllPotential(N_points)
Tk_sqr, densx_sqr = xsolver(N_points, H_sqr, N_electron)

X = np.linspace(0, 1, N_points, endpoint=True)

# harmonic oscillator
real_density_1 = np.sqrt(omega/np.pi)*np.exp(-omega*(X-0.5)**2)
real_density_2 = 2*np.sqrt(omega/np.pi)*omega*((X-0.5)**2)*np.exp(-omega*(X-0.5)**2)
two_electron = real_density_1 + real_density_2

# kinetic energy 
real_density = 2*(np.sin(np.pi*X))**2 + 2*(np.sin(2*np.pi*X))**2
real_KineticEnergy = np.pi**2/2 + 2*np.pi**2

print(real_KineticEnergy)
print(Tk_sqr)

plt.plot(X, two_electron, 'r')
plt.plot(X, densx_har, 'b-.')
plt.show()