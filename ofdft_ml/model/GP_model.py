import numpy as np 

from ofdft_ml.statslib.pca import PrincipalComponentAnalysis as PCA 
from ofdft_ml.statslib.GaussProcess import GaussProcessRegressor as GPR 
from ofdft_ml.statslib.kernel import rbf_kernel, rbf_kernel_gradient

__all__ = ['GP_model']

class GP_model(object):
    def __init__(self, n_components=2, gamma=1e-3, beta=1e-8):
        self.n_cmp = n_components
        self.gamma = gamma
        self.beta = beta

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if (value <= 1) and (value >= 1e-8):
            self._gamma = value 
        else:
            print('gamma value should range from 1e-8 to 1')
    
    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if (value <= 1e-5) and (value >= 1e-12):
            self._beta = value
        else:
            print('beta value should range from 1e-12 to 1e-5')
    
    def fit(self, X, y):
        pca = PCA(self.n_cmp)
        tX = pca.fit_transform(X)
        estimator = GPR(gamma=self.gamma, beta=self.beta, kernel=rbf_kernel,
                        kernel_gd=rbf_kernel_gradient,
                        optimize=True,
                        params_bounds=((1e-8, 1), (1e-12, 1e-5)))
        estimator.fit(tX, y)
        print('Finish fitting ......')

        self.transformer = pca
        self.estimator = estimator

    def predict(self, X):
        tX = self.transformer.transform(X)
        pred_y = self.estimator.predict(tX)
        return pred_y

    def predict_gradient(self, X):
        tX = self.transformer.transform(X)
        pred_X_dy = self.transformer.inverse_transform_gradient(self.estimator.predict_gradient(tX))
        return pred_X_dy

