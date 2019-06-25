import numpy as np
from sklearn.model_selection import KFold
from copy import deepcopy

class Trainer(object):
    def __init__(self, dataset, model, metrics, n_cv, transformer=None):
        """
        dataset: a Dataset object
        transformer: PCA data transform
        model: ML model
        metrics: list of metrics
        """
        self.dataset = dataset
        self.default_transformer = transformer
        self.default_model = model
        self.metrics = metrics
        self.n_cv = n_cv

    def train(self):
        if self.default_transformer is not None:
            self.n_cmps = self.default_transformer.n_cmps
        else:
            self.n_cmps = None

        self.summary = {'n_components': self.n_cmps}

        data_index = np.arange(0, self.dataset.len_train, 1, int)
        kf = KFold(self.n_cv)
        for metric in self.metrics:
            tmp_summary = {'hyperparameter': [self.default_model.gamma, self.default_model.noise], 
                           'scalar_error': np.inf,
                           'vector_error': np.inf}
            total_error = np.inf
            for train_index, valid_index in kf.split(data_index):
                train_features, train_targets = self.dataset.all_train_features[train_index], self.dataset.all_train_targets[train_index]
                valid_features, valid_targets = self.dataset.all_train_features[valid_index], self.dataset.all_train_targets[valid_index]
                model = deepcopy(self.default_model)
                transformer = deepcopy(self.default_transformer)

                if transformer is not None:
                    print('Doing data transformation !')
                    train_data = {'features': train_features, 'targets': train_targets}
                    transformed_train_data = transformer.fit_transform(train_data)
                    train_features = transformed_train_data['features']
                    train_targets = transformed_train_data['targets']

                    valid_data = {'features': valid_features, 'targets': valid_targets}
                    transformed_valid_data = transformer.transform(valid_data)
                    valid_features = transformed_valid_data['features']
                    valid_targets = transformed_valid_data['targets']
                
                model.fit(train_features, train_targets)
                predict_targets = model.predict(valid_features)

                errors = metric(predict_targets, valid_targets)
                new_total_error = errors['scalar_error'] + errors['vector_error']
                if new_total_error < total_error:
                    tmp_summary.update({
                        'hyperparameter': [model.gamma, model.noise],
                        'scalar_error': errors['scalar_error'],
                        'vector_error': errors['vector_error']
                    })
                    total_error = new_total_error
                
            self.summary[metric.__name__] = tmp_summary