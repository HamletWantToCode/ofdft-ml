import numpy as np

class Trainer(object):
    def __init__(self, dataset, model, transformer=None , metrics=None):
        """
        dataset: a Dataset object
        transformer: PCA data transform
        model: ML model
        metrics: list of metrics
        """
        self.dataset = dataset
        self.transformer = transformer
        self.model = model
        self.metrics = metrics

        self.transformation = None

    def train(self):
        self.train_data = {'features': self.dataset.train_features,
                           'targets': self.dataset.train_targets}
        if callable(self.transformer):
            transformed_train_data = self.transformer(self.train_data)
            features = transformed_train_data['features']
            targets = transformed_train_data['targets']
            self.transformation = self.transformer.tr_mat
        else:
            features = self.train_data['features']
            targets = self.train_data['targets']
        self.model.fit(features, targets)
        # train results
        self.hyperparameters = [self.model.gamma, self.model.noise]
        self.summary = {'transformation': self.transformation,
                        'hyperparameters': self.hyperparameters}

        if self.metrics is not None:
            err_measures = self.validate()
            self.summary.update(err_measures)

    def validate(self):
        self.validate_data = {'features': self.dataset.valid_features,
                              'targets': self.dataset.valid_targets}
        if callable(self.transformer):
            transformed_valid_data = self.transformer(self.validate_data)
            features = transformed_valid_data['features']
            targets = transformed_valid_data['targets']
        else:
            features = self.validate_data['features']
            targets = self.validate_data['targets']
        predicts = self.model.predict(features)

        results = {}
        for metric in self.metrics:
            results[metric.__name__] = metric(predicts, targets)
        return results




