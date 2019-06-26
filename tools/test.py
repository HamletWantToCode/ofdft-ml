import numpy as np
import matplotlib.pyplot as plt
import json
import os
from ofdft_ml.statslib import Model_loader

# dataset
data_dir = 'datasets/multitask/'
# parameter
param_dir = 'summary/'
flist = os.listdir(param_dir)
flist.sort()
param_fname = param_dir + flist[-1]

loader = Model_loader(data_dir+'train/', param_fname)
ml_model = loader.load()

# predict on testing dataset
test_feature = np.load(data_dir + 'test/features.npy')
test_targets = np.load(data_dir + 'test/targets.npy')
test_data = {'features': test_feature, 'targets': test_targets}

pred_Ek, pred_Ek_derivative = ml_model(test_feature)

# plots
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(test_targets[:, 0], pred_Ek, 'bo')
ax1.plot(test_targets[:, 0], test_targets[:, 0], 'r')

X = np.linspace(0, 1, 500)
ax2.plot(X, test_targets[34, 1:], 'r', X, pred_Ek_derivative[34], 'b')
plt.show()
