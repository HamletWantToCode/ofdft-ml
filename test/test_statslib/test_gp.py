# test gauss process regressor in 1D

import numpy as np 

from statslib.gauss_process import GaussProcess 
from statslib.kernel_ridge import KernelRidge
from statslib.utils import rbf_kernel, rbf_kernel_gradient, rbf_kernel_hessan

gamma = 0.1

def kernel(gamma, x, y):
    dx = x - y.T
    return np.exp(-gamma*dx**2)

def kernel_gd(gamma, x, y):
    dx = x - y.T
    return -2*gamma*dx*np.exp(-gamma*dx**2)
    
def kernel_hess(gamma, x, y):
    dx = x - y.T
    return 2*gamma*(1-2*gamma*dx**2)*np.exp(-gamma*dx**2) 

def f(x):
    return x*np.sin(x)

def df(x):
    return np.sin(x) + x*np.cos(x)

X = np.arange(0.25*np.pi, 2*np.pi, 0.5*np.pi)
y = f(X)
dy = df(X)
# dy = np.zeros(len(X))

krr = KernelRidge(gamma=gamma, kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient)
krr.fit(X[:, np.newaxis], y)

gp = GaussProcess(gamma=gamma, kernel=rbf_kernel, gradient_on=True, kernel_gd=rbf_kernel_gradient, kernel_hess=rbf_kernel_hessan)
gp.fit(X[:, np.newaxis], y, dy[:, np.newaxis])

Xt = np.linspace(0, 2*np.pi, 50)
## standard
yt = f(Xt)
dyt = df(Xt)
## gauss process
gp_y_pred = gp.predict(Xt[:, np.newaxis])
gp_y_var = gp.predict_variance(Xt[:, np.newaxis])
gp_y_gd = gp.predict_gradient(Xt[:, np.newaxis])
## kernel ridge regression
krr_y_pred = krr.predict(Xt[:, np.newaxis])
krr_y_gd = krr.predict_gradient(Xt[:, np.newaxis])

import matplotlib.pyplot as plt 
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(X, y, 'ko', label='train data')
ax1.plot(Xt, yt, 'r--', linewidth=2, label='True curve')
ax1.plot(Xt, gp_y_pred, 'b', alpha=0.7, label='Gauss process')
ax1.fill_between(Xt, gp_y_pred-gp_y_var, gp_y_pred+gp_y_var, color='b', alpha=0.5)
ax1.plot(Xt, krr_y_pred, 'g', alpha=0.7, label='kernel ridge regression')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()

ax2.plot(Xt, dyt, 'r--', linewidth=2, label='True derivative')
ax2.plot(Xt, gp_y_gd, 'b', alpha=0.7, label='Gauss process')
ax2.plot(Xt, krr_y_gd, 'g', alpha=0.7, label='kernel ridge regression')
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\frac{df(x)}{dx}$')
ax2.legend()

plt.savefig('test/test_statslib/gp_vs_krr.png')
