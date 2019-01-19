# constrained optimization with Newton's method

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def f(x):
    return 0.5*(x[0]**2 + x[1]**2)

def df(x):
    return np.array([x[0], x[1]])

def inv_ddf(x):
    return np.array([[1, 0],
                     [0, 1]])

def g(x):
    return x[0]+x[1]-1

def dg(x):
    return np.array([1, 1])

x0 = np.array([1, 0])
mu = -0.5
tol = 1e-3
n = 1
eta = 0.5
maiters = 100
func0 = f(x0) + mu*g(x0)
while True:
    print(x0)
    gd = df(x0) + mu*dg(x0)
    x = x0 - eta*(inv_ddf(x0) @ gd)
    func1 = f(x) + mu*g(x)    
    err = abs(func0 - func1)
    x0 = x
    func0 = func1
    if err < tol:
        print('convergency reached at %d step' %(n))
        break
    if n > maiters:
        print('loop exceed')
        break
    n += 1

fig = plt.figure()
ax = fig.gca()
X = Y = np.linspace(-1, 1, 50)
XX, YY = np.meshgrid(X, Y)
XY = product(X, Y)
Z = np.array([f(xy) for xy in XY]).reshape(XX.shape)

ax.contourf(XX, YY, Z, 30)
ax.plot(X, 1-X, 'r--')
ax.plot(x0[0], x0[1], 'ko')
plt.show()


