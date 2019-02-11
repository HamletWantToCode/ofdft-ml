# demo of potential function 

import numpy as np 
import matplotlib.pyplot as plt 

from quantum.utils import simple_potential_gen

nbasis = 10
low_V0, high_V0 = 1, 5
low_Phi0, high_Phi0 = -0.15, 0.15
params_gen = simple_potential_gen(nbasis, low_V0, high_V0, low_Phi0, high_Phi0, 3892)

fig = plt.figure()
ax = fig.gca()
X = np.linspace(0, 1, 100)
for i in range(10):
    _, Vq = next(params_gen)
    Vx = np.fft.irfft(Vq, 100)*100
    ax.plot(X, Vx)
ax.set_xlabel('x')
ax.set_ylabel('V(x)')
plt.savefig('test/test_quantum/potential_demo.png')