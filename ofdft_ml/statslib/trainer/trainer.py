import numpy as np
from sklearn.model_selection import KFold
from copy import deepcopy
from joblib import Parallel, delayed

class Trainer(object):
    def __init__(self, dataset, model, metrics, n_cv, transformer=None, n_jobs=1):
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
        self.n_jobs = n_jobs

    def train(self, start_ups):
        if self.default_transformer is not None:
            self.n_cmps = self.default_transformer.n_cmps
        else:
            self.n_cmps = None

        self.summary = {'n_components': self.n_cmps}

        data_index = np.arange(0, self.dataset.len_train, 1, int)
        kf = KFold(self.n_cv)
        init_gammas, init_noises = np.meshgrid(start_ups['gamma'], start_ups['noise'])
        hyperparameters = zip(init_gammas.ravel(), init_noises.ravel())

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=2*self.n_jobs)

        with parallel:
            for metric in self.metrics: 
                mean_total_error = np.inf
                tmp_summary = {}

                for gamma0, noise0 in hyperparameters:
                    model0 = deepcopy(self.default_model)
                    model0.gamma = gamma0
                    model0.noise = noise0
            
                    results = parallel(
                                        delayed(_cross_validate_score)
                                        (
                                            deepcopy(model0), 
                                            deepcopy(self.default_transformer),
                                            metric,
                                            self.dataset.all_train_features,
                                            self.dataset.all_train_targets,
                                            train_index, 
                                            valid_index,
                                        )
                                        for train_index, valid_index in kf.split(data_index)
                                        )
                    results = np.array(results)
                    
                    mean_scalar_error, mean_vector_error = np.mean(results, axis=0)
                    mean_total_error_ = mean_scalar_error + mean_vector_error

                    if mean_total_error_ < mean_total_error:
                        tmp_summary.update({
                            'starting_point': [gamma0, noise0],
                            'scalar_error': mean_scalar_error,
                            'vector_error': mean_vector_error,
                        })
                        mean_total_error = mean_total_error_

                    self.summary[metric.__name__] = tmp_summary

def _cross_validate_score(model, transformer, metric, X, y, train_index, valid_index):
    train_features, train_targets = X[train_index], y[train_index]
    valid_features, valid_targets = X[valid_index], y[valid_index]

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
    return errors['scalar_error'], errors['vector_error']