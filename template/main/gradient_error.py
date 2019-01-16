# plot gradient prediction error

import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator 
from matplotlib.gridspec import GridSpec

from statslib.pca import PrincipalComponentAnalysis
from statslib.kernel_ridge import KernelRidge
from statslib.pipeline import MyPipe
from statslib.utils import rbf_kernel, rbf_kernel_gradient

with open('../results/demo_train_data', 'rb') as f:
    train_data = pickle.load(f)
with open('../results/demo_test_data', 'rb') as f1:
    test_data = pickle.load(f1)
Ek_train, densx_train, dEkx_train = train_data[:, 0], train_data[:, 1:503], train_data[:, 503:]
Ek_test, densx_test, dEkx_test = test_data[:, 0], test_data[:, 1:503], test_data[:, 503:]

pca = PrincipalComponentAnalysis()
C, gamma = 1.09854114e-10, 0.0009102982
krr = KernelRidge(C=C, gamma=gamma, kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient)
pipe = MyPipe([('reduce_dim', pca), ('regressor', krr)])

Error = []
for i in range(20):
    pipe.set_params(reduce_dim__n_components=i+1)
    pipe.fit(densx_train, Ek_train)
    dEkxt_predict = pipe.predict_gradient(densx_test)
    dEkx_predict = pipe.named_steps['reduce_dim'].inverse_transform_gradient(dEkxt_predict)
    Error.append(np.mean(np.mean((dEkx_predict-dEkx_test)**2, axis=1)))

fig = plt.figure(figsize=(10, 4))
gd = GridSpec(1, 2, fig)
ax = fig.add_subplot(gd[0])
ax.plot(np.arange(1, 21, 1), Error, 'bo-')
ax.set_xlabel('# principal component')
ax.xaxis.set_major_locator(MultipleLocator())
ax.set_ylabel('mean square error')

ax2 = fig.add_subplot(gd[1])
X = np.linspace(0, 1, 502)
for i in [1, 4, 7, 10, 13]:
    pipe.set_params(reduce_dim__n_components=i)
    pipe.fit(densx_train, Ek_train)
    dEkxt_predict = pipe.predict_gradient(densx_test[0][np.newaxis, :])
    dEkx_predict = pipe.named_steps['reduce_dim'].inverse_transform_gradient(dEkxt_predict)
    ax2.plot(X, dEkx_predict[0], linestyle='--', alpha=(1-0.05*i), label='#%d' %(i))
ax2.plot(X, dEkx_test[0], 'r', label='true')
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\hat{P}\left[\frac{\delta T}{\delta n(x)}\right]$')
ax2.legend()
plt.savefig('../results/gradient_error.png')