import numpy as np 
import json 
from ofdft_ml.statslib import Forward_PCA_transform, Backward_PCA_transform, ScalarGP, MultitaskGP, SeqModel

class Model_loader(object):
    def __init__(self, train_data_dir, param_fname):
        train_features = np.load(train_data_dir + 'features.npy')
        train_targets = np.load(train_data_dir + 'targets.npy')
        with open(param_fname, 'r') as f:
            parameters = json.load(f)
        n_components = parameters['n_components']
        gamma, noise = parameters['mean_square_error']['hyperparameter']

        self.train_data = {'features': train_features, 'targets': train_targets}
        self.n_cmps = n_components
        self.gamma = gamma
        self.noise = noise

    def load(self):
        forward_transformer = Forward_PCA_transform(self.n_cmps)
        tr_train_data = forward_transformer.fit_transform(self.train_data)
        tr_train_features = tr_train_data['features']
        tr_train_targets = tr_train_data['targets']

        regressor = SeqModel(self.gamma, self.noise, ((1e-4, 5.0), (1e-5, 0.1)), ScalarGP, MultitaskGP, 'eval')
        regressor.fit(tr_train_features, tr_train_targets)

        backward_transformer = Backward_PCA_transform(forward_transformer)

        chained_model = Composer(forward_transformer, regressor, backward_transformer)
        return chained_model


class Composer(object):
    def __init__(self, forward_transformer, model, backward_transformer):
        self.forward_transformer = forward_transformer
        self.model = model
        self.backward_transformer = backward_transformer

    def __call__(self, feature):
        if feature.ndim < 2:
            feature = feature[np.newaxis, :]
        tr_feature = self.forward_transformer.transform_x(feature)
        pred_targets =self.model.predict(tr_feature)
        Ek = pred_targets[:, 0]
        Ek_derivative = self.backward_transformer(pred_targets[:, 1:])
        return Ek, Ek_derivative


    