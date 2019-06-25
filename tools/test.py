import numpy as np
import matplotlib.pyplot as plt
import json
from ofdft_ml.statslib import Model_loader

# dataset
dir_name = 'datasets/multitask/train/'
# parameter
param_name = 'summary/train_at_2019-06-25_14:45:51'

loader = Model_loader(dir_name, param_name)
ml_model = loader.load()

# predict on testing dataset
test_feature = np.load(dir_name + 'features.npy')
test_targets = np.load(dir_name + 'targets.npy')
test_data = {'features': test_feature, 'targets': test_targets}

pred_Ek, pred_Ek_derivative = ml_model(test_feature)

# plots
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(test_targets[:, 0], pred_Ek, 'bo')
ax1.plot(test_targets[:, 0], test_targets[:, 0], 'r')

X = np.linspace(0, 1, 500)
ax2.plot(X, test_targets[34, 1:], 'r', X, pred_Ek_derivative[34], 'b')
plt.show()
