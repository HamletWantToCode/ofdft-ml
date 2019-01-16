# constrained optimization

import numpy as np
import matplotlib.pyplot as plt

mu = 2
eta = 1e-2
err = 1e-4

def line(t):
    return np.array([np.cos(t), np.sin(t)])

def f_origin(x):
    return x[0] + x[1]

def f(x, mu):
    return x[0] + x[1] - mu*(x[0]**2 + x[1]**2 - 1)

def df(x, mu):
    return np.array([1-2*mu*x[0], 1-2*mu*x[1]])

x0 = np.array([-1, 0])
f0 = f(x0, mu)
for i in range(1000):
    gd = df(x0, mu)
    x = x0 - eta*gd
    f1 = f(x, mu)
    if abs(f1 - f0)<err:
        print('convergency reached after %d of steps' %(i))
        x0 = x
        print(x0)
        break
    x0 = x
    f0 = f1

X = Y = np.linspace(-1, 1, 20)
xx, yy = np.meshgrid(X, Y)
Params = np.c_[xx.reshape((-1, 1)), yy.reshape((-1, 1))]
zz = np.array([f_origin(xy) for xy in Params]).reshape(xx.shape)
# theta = np.linspace(0, 2*np.pi, 30)

plt.contourf(xx, yy, zz, 30)
plt.plot(x0[0], x0[1], 'ko')
# plt.plot(X, y, 'r')
plt.ylim([-1, 1])
plt.show()

