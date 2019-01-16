import numpy as np 
from sklearn.utils import check_array

class Minimizer(object):
    def __init__(self, ml_model):
        self.transformer = ml_model.named_steps['reduce_dim']
        self.estimator = ml_model.named_steps['regressor']

    def energy(self, dens, V, mu, N):
        _, n_points = dens.shape
        denst = self.transformer.transform(dens)
        Ek = self.estimator.predict(denst)[0]
        return Ek + np.sum((V[0] - mu)*dens[0])*(1.0/n_points) + mu*N 

    def energy_gd(self, dens, V, mu):
        _, n_points = dens.shape
        denst = self.transformer.transform(dens)
        dEkt = (n_points-1) * self.estimator.predict_gradient(denst)
        dEk = self.transformer.inverse_transform_gradient(dEkt)[0]
        return dEk + V - mu

    def run(self, dens_init, V, mu, N, eta=1e-3, err=1e-2, maxiters=1000, verbose=False):
        dens_init, V = check_array(dens_init), check_array(V)

        E0 = self.energy(dens_init, V, mu, N)
        n = 1
        while True:
            gd = self.energy_gd(dens_init, V, mu)
            dens = dens_init - eta*gd
            E1 = self.energy(dens, V, mu, N)
            dens_init = dens
            if verbose and (n%10==0):
                print('Previous E0=%.5f, After %d iteration, E1=%.5f, |E|=%.5f' %(E0, n, E1, abs(E1-E0)))
            if abs(E1 - E0)<err:
                print('converge after %d of iterations !' %(n))
                break
            E0 = E1
            n += 1
            if n > maxiters:
                raise StopIteration('exceed maximum iteration number !')
        return dens_init[0]