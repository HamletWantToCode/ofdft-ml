from ofdft_ml.statslib import *
import json
import time

# dataset settings
data_dir = 'datasets/multitask/'
test_size = 0.2

dataset = Dataset(data_dir, test_size)
dataset.train_test()

# model setting
gamma = 0.1
noise = 0.01
bounds = ((1e-4, 5.0), (1e-5, 0.1))

# model = ScalarGP(gamma, noise, bounds)
model = SeqModel(gamma, noise, bounds, ScalarGP, MultitaskGP, mode='train')

# train setting
pca_components = 5
transformer = Forward_PCA_transform(pca_components)
metrics = [mean_square_error]
trainer = Trainer(dataset, model, metrics, 5, transformer)
trainer.train()

# results summary
datetime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
summary_fname = 'summary/train_at_' + datetime
with open(summary_fname, 'w') as f:
    json.dump(trainer.summary, f)

