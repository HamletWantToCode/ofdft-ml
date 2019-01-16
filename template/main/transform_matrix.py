# plot the transformation matrix

import numpy as np 
import pickle
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid

with open('../results/demo_best_estimator', 'rb') as f:
    estimator = pickle.load(f)

pca = estimator.named_steps['reduce_dim']
trans_mat = pca.tr_mat_

fig = plt.figure() 
gs = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1, aspect=False)
for i in range(4):
    gs[i].plot(np.arange(0, 502, 1), trans_mat[:, i], 'b', label='#%d' %(i+1))
    gs[i].legend()

fig.text(0.5, 0.01, 'x points')
fig.text(0.01, 0.5, 'matrix value', va='center', rotation='vertical')
plt.savefig('../results/demo_transform_matrix.png')