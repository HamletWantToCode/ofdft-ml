import numpy as np 
from sklearn.utils import check_array

class EulerSolver(object):
    def __init__(self, model):
        self.tr_mat = model.transformer.tr_mat_
        self.model = model

    def energy(self, dens, V, mu, N):
        _, n_points = dens.shape
        Ek = self.model.predict(dens)[0]
        return Ek + np.sum((V - mu)*dens[0])*(1.0/n_points) + mu*N 

    def energy_gd(self, dens, V, mu):
        dEk = self.model.predict_gradient(dens)
        V_proj = (V-mu) @ self.tr_mat @ self.tr_mat.T
        return dEk + V_proj

    def run(self, dens_init, V, mu, N, eta=0.1, err=1e-8, maxiters=1000, verbose=False):
        dens_init = check_array(dens_init)

        E0 = self.energy(dens_init, V, mu, N)
        n = 1
        while True:
            gd = self.energy_gd(dens_init, V, mu)
            dens = dens_init - eta*gd
            E1 = self.energy(dens, V, mu, N)
            assert E1 < E0, print('In gradient descent, E should decrease in each step')
            dens_init = dens
            if verbose and (n%10==0):
                print('Previous E0=%.5f, After %d iteration, E1=%.5f, |E|=%.5f' %(E0, n, E1, abs(E1-E0)))
            if abs(E1 - E0)<err:
                print('converge after %d of iterations !' %(n))
                break
            E0 = E1
            n += 1
            if n > maxiters:
                print('loop exceed')
                break
                # raise StopIteration('exceed maximum iteration number !')
        return dens_init[0]