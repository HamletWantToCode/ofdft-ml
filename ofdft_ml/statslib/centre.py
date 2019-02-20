import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

class Centre(BaseEstimator, TransformerMixin):
    def __init__(self, mid=None):
        self.mid = mid

    def fit(self, X, y=None):
        X = check_array(X)
        if self.mid is None:
            self.mid = X.shape[1] // 2
        peak_pos = self.mid - np.argmax(abs(X), axis=1)
        self.shift_ = peak_pos
    
    def transform(self, X):
        X = check_array(X)
        n_samples = len(X)
        X_ = np.zeros_like(X)
        for i in range(n_samples):
            X_[i] = np.roll(X[i], shift=self.shift_[i])
        return X_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform_gradient(self, dy):
        return self.transform(dy)