# finite difference method

import numpy as np

def xsolver(n_points, hamilton_mat, ne):
    _, Psi = np.linalg.eigh(hamilton_mat)
    # compute electron density
    density = np.zeros(n_points)
    for i in range(ne):
        density[1:-1] += (n_points-1)*(Psi[:, i] * Psi[:, i].conj()).real
    # compute kinetic energy
    T = np.zeros((n_points-2, n_points-2))
    np.fill_diagonal(T, -2)
    np.fill_diagonal(T[1:, :-1], 1)
    np.fill_diagonal(T[:-1, 1:], 1) 
    Tk = 0
    for i in range(ne):
        Tk += (Psi[:, i].conj() @ T @ Psi[:, i]).real
    Tk *= -0.5*(n_points-1)**2
    return Tk, density

# def finiteDifferenceMatrix(n, xstart, xend, potentialFunction):
#     T = np.zeros((n, n), np.float64)
#     V = np.zeros((n, n), np.float64)
#     L = xend - xstart
#     np.fill_diagonal(T, -2)
#     np.fill_diagonal(T[1:, :-1], 1)
#     np.fill_diagonal(T[:-1, 1:], 1)
#     h = L*1.0/(n+1)
#     Vx = [potentialFunction(xstart+h*(i+1)) for i in range(n)]
#     np.fill_diagonal(V, Vx)
#     return T*(-0.5*(n+1)**2)/L**2 + V, Vx

# def electronDensity(Psi, ne, n, xstart, xend):
#     # not consider spin
#     density = np.zeros(n, np.float64)
#     L = xend - xstart
#     for i in range(ne):
#         density += ((n+1)*1.0/L)*(Psi[:, i]**2)
#     return np.r_[0, density, 0]

# def kineticEnergy(n, xstart, xend, ne, Psi):
#     Tk = np.zeros((n, n+2), np.float64)
#     L = xend - xstart
#     Ek = 0
#     for i in range(n):
#         Tk[i, i], Tk[i, i+1], Tk[i, i+2] = 1, -2, 1
#     Tk *= -0.5*(n+1)**2/L**2
#     for j in range(ne):
#         Psi_j_withEndPoints = np.r_[0, Psi[:, j], 0]
#         Ek += Psi[:, j] @ Tk @ Psi_j_withEndPoints
#     return Ek

# def compute(n, ne, A, B, xstart=0, xend=1):
#     potential = sinPotential(A, B)
#     H, Vx = finiteDifferenceMatrix(n, xstart, xend, potential)
    
#     density = electronDensity(eigenFunctions, ne, n, xstart, xend)
#     Ek = kineticEnergy(n, xstart, xend, ne, eigenFunctions)
#     return np.array([ne, Ek, *density]), np.array([ne, 0, *Vx, 0])








