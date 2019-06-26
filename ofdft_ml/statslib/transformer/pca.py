# Principal Component Analysis
import numpy as np

class Forward_PCA_transform(object):
    def __init__(self, n_cmps=2):
        self.n_cmps = n_cmps
        self.tr_mat = None

    def fit(self, X):
        self._mean = np.mean(X, axis=0)
        center_X = X - self._mean
        U, S, _ = np.linalg.svd(center_X.T)
        self.tr_mat = U[:, :self.n_cmps]
        self.explain_rate = np.sum(S[:self.n_cmps]**2) / np.sum(S**2)
    
    def transform(self, data):
        X = data['features']
        all_targets = data['targets']
        if all_targets.ndim > 1:
            DTDX = data['targets'][:, 1:]
            Ek = data['targets'][:, 0]
        else:
            Ek = all_targets
        
        Xt = (X-self._mean) @ self.tr_mat
        
        if all_targets.ndim > 1:
            DTDXt = DTDX.dot(self.tr_mat)
            targets_ = np.c_[Ek[:, np.newaxis], DTDXt]
        else:
            targets_ = Ek
        data_ = {'features': Xt, 'targets': targets_}
        return data_

    def transform_x(self, features):
        return (features - self._mean) @ self.tr_mat

    def fit_transform(self, data):
        X = data['features']
        self.fit(X)
        return self.transform(data)


class Backward_PCA_transform(object):
    def __init__(self, forward_transformer):
        self.back_tr_mat = forward_transformer.tr_mat.T
        
    def __call__(self, targets):
        return targets @ self.back_tr_mat