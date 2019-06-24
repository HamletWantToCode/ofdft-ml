# Principal Component Analysis
import numpy as np

class Forward_PCA_transform(object):
    def __init__(self, n_cmps=2, explain_rate=None):
        self.n_cmps = n_cmps
        if explain_rate is not None:
            self.n_cmps = None
            self.explain_rate = explain_rate
        self.tr_mat = None

    def fit(self, X):
        self._mean = np.mean(X, axis=0)
        center_X = X - self._mean
        U, S, _ = np.linalg.svd(center_X.T)
        if self.n_cmps is not None:
            self.tr_mat = U[:, :self.n_cmps]
        else:
            var = S**2
            s = 0
            total_var = np.sum(var)
            for i, var_i in enumerate(var):
                s += var_i
                if (s / total_var) > self.explain_rate:
                    break
            self.tr_mat = U[:, :i]
    
    def transform(self, X):
        if self.tr_mat is not None:
            return (X-self._mean) @ self.tr_mat
        else:
            raise ValueError

    def __call__(self, data):
        X = data['feature']
        DTDX = data['targets'][:, 1:]
        Ek = data['targets'][:, 0]
        if self.tr_mat is None:
            self.fit(X)
        Xt = self.transform(X)
        DTDXt = DTDX.dot(self.tr_mat)
        targets_ = np.c_[Ek[:, np.newaxis], DTDXt]
        data_ = {'feature': Xt, 'targets': targets_}
        return data_