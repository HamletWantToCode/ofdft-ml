# Principal Component Analysis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
import numpy as np 

class PrincipalComponentAnalysis(BaseEstimator, TransformerMixin):
    """ 
    Parameters
    ----------
    n_components:

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        mean = np.mean(X, axis=0)
        Cov = (X - mean).T.conj() @ (X - mean)
        U, _, _ = np.linalg.svd(Cov)
        tr_mat = U[:, :self.n_components]
        self.mean_ = mean
        self.tr_mat_ = tr_mat
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        """ 
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        X = check_array(X)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        tr_X = (X - self.mean_) @ self.tr_mat_
        return tr_X

    def transform_gradient(self, dy):
        dy = check_array(dy)
        if dy.shape[1] != self.n_features_:
                raise ValueError('Shape of input is different from what was seen'
                                'in `fit`') 
        tr_dy = dy @ self.tr_mat_
        return tr_dy

    def inverse_transform(self, Xt):
        X = Xt @ self.tr_mat_.T + self.mean_
        return X

    def inverse_transform_gradient(self, dyt):
        dy = dyt @ self.tr_mat_.T 
        return dy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
