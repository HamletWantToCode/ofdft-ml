import numpy as np
import json

class OFDFT(object):
    def __init__(self, KE_functional, n_particle):
        self.KE_func = KE_functional
        self.n_particle = n_particle

    def __call__(self, x, potential):
        max_iters = 1000
        gtol = 1e-5
        eta = 0.01

        _log = {}
        _log['optimization setting'] = {'max_iters': max_iters,
                                        'step': eta,
                                        'gtol': gtol}

        func_val = []
        for i in range(max_iters):
            E = self.energy(x, potential)
            func_val.append(E)
            grad = self.energy_gradient(x, potential)
            norm_grad = np.sqrt((1.0/len(x[:-1])))*np.linalg.norm(grad)
            if norm_grad < gtol:
                _log['success'] = 'YES'
                print('optimization converge after %d-steps!' %(i+1))
                break
            x -= eta*grad
        if i==(max_iters-1):
            _log['success'] = 'NO'
            print('optimization failed !')
        _log['history'] = func_val

        with open('optim_log', 'w') as f:
            json.dump(_log, f)

        return x[:-1]

    def energy(self, x, Vx):
        rho = x[:-1]
        mu = x[-1]
        Ek, _ = self.KE_func(rho)
        return Ek[0] + (1.0/len(rho))*np.sum((Vx-mu)*rho) + mu*self.n_particle

    def energy_gradient(self, x, Vx):
        rho = x[:-1]
        mu = x[-1]
        if hasattr(self.KE_func, 'forward_transformer'):
            tr_mat = self.KE_func.forward_transformer.tr_mat
            Vxt = (Vx[np.newaxis, :] - mu) @ tr_mat
            Vx_ = Vxt @ tr_mat.T
        else:
            Vx_ = (Vx - mu)[np.newaxis, :]
        _, Ek_derivative = self.KE_func(rho)
        E_derivative = Ek_derivative[0] + Vx_[0]
        return np.array([*E_derivative, self.n_particle-(1.0/len(rho))*np.sum(rho)])

# from scipy.optimize import minimize, LinearConstraint, BFGS, Bounds
# A = np.ones_like(rho)*(1.0/len(rho))
# linear_constraint = LinearConstraint([A], self.n_particle, self.n_particle)
# bounds = Bounds(0, np.inf)
# results = minimize(self.energy, rho, args=(potential),
#                    method='trust-constr',
#                    jac=self.energy_gradient, hess=BFGS(),
#                    constraints=[linear_constraint],
#                    options={'verbose': 1}, bounds=bounds)
# return results.x
