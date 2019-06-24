from ofdft_ml.statslib import Dataset, ScalarGP, MultitaskGP, Forward_PCA_transform
from ofdft_ml.statslib import Trainer
import pickle
import json
import time

# dataset settings
data_dir = 'some_location'
test_size = 0.2
valid_size = 0.2

dataset = Dataset(data_dir, test_size, valid_size)
dataset.train_test()
dataset.train_validate()

# model setting
pca_components = 7
gamma = 0.1
noise = 1e-3

transformer = Forward_PCA_transform(pca_components)
model = SeqModel(gamma, noise, bounds, ScalarGP, MultitaskGP)

# train setting
trainer = Trainer(dataset, transformer, model, metric)
trainer.train()

# results summary
datetime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
fname = 'train_at_' + datetime
with open(fname, 'w') as f:
    json.dump(trainer.summary, f)
model_fname = 'model_at_' + datetime
with open(model_fname, 'wb') as mf:
    pickle.dump(model, mf)
transformer_fname = 'transformer_at_' + datetime
with open(transformer_fname, 'wb') as tf:
    pickle.dump(transformer, tf)
