# Principal Component Analysis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
import numpy as np

class PrincipalComponentAnalysis(BaseEstimator, TransformerMixin):
    """
    Embedding the high dimensional data to a low dimensional hyperplane,

    :param n_components: int, desired dimension for reduction

    Examples::

    >>> from statslib.pca import PrincipalComponentAnalysis as PCA
    >>> import numpy as np
    >>> X = np.random.rand(10, 8)
    >>> pca = PCA()
    >>> Xt = PCA.fit_transform(X)
    >>> assert Xt.shape[1]==2

    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        """
        fit the original data, compute transformation matrix.

        :param X: array, shape (n_samples, n_features), feature of training data

        :param y: None, There is no need of a target in a transformer, yet the pipeline API requires this parameter.

        :return self: object
        """
        X = check_array(X)
        mean = np.mean(X, axis=0)
        Cov = (X - mean).T.conj() @ (X - mean)
        U, S, _ = np.linalg.svd(Cov)
        total_var = np.sum(S)
        explained_var_ratio = S[:self.n_components] / total_var
        tr_mat = U[:, :self.n_components]
        self.mean_ = mean
        self.tr_mat_ = tr_mat
        self.n_features_ = X.shape[1]
        self.explained_var_ratio = explained_var_ratio
        return self

    def transform(self, X):
        """
        transform the feature from original space to a low dimensional space

        :param X: array, shape (n_samples, n_features), feature in original space

        :return tr_X: array, shape (n_samples, n_components), feature in low dimensional space
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
        """
        apply same transformation on the function gradient, transform it into the low dimensional space

        :param dy: array, shape (n_samples, n_features), function gradient in original space

        :return tr_dy: array, shape (n_samples, n_components), transformed function gradient
        """
        dy = check_array(dy)
        if dy.shape[1] != self.n_features_:
                raise ValueError('Shape of input is different from what was seen'
                                'in `fit`')
        tr_dy = dy @ self.tr_mat_
        return tr_dy

    def inverse_transform(self, Xt):
        """
        apply inverse transform, and transform features back to the original space

        :param Xt: array, shape (n_samples, n_components), feature in low dimensional space

        :return X: array, shape (n_samples, n_features), feature in original space
        """
        X = Xt @ self.tr_mat_.T + self.mean_
        return X

    def inverse_transform_gradient(self, dyt):
        """
        transform the gradient back to the original space

        :param dyt: array, shape (n_samples, n_components), gradient in low dimensional space

        :return dy: array, shape (n_samples, n_features), gradient in original space
        """
        dy = dyt @ self.tr_mat_.T
        return dy

    def fit_transform(self, X, y=None):
        """
        combine the ``fit`` and ``transform`` method

        :param X: array, shape (n_samples, n_features)
        :param y: None

        :return Xt: array, shape (n_samples n_components)
        """
        self.fit(X, y)
        return self.transform(X)
