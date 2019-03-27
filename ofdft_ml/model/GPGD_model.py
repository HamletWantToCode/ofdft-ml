import numpy as np 
from ofdft_ml.statslib.pca import PrincipalComponentAnalysis as PCA
from ofdft_ml.statslib.GaussProcess import GaussProcessRegressor as GPR
from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_2nd_gradient 

__all__ = ['GPGD_model']

class GPGD_model(object):
    def __init__(self, n_components=2, gamma=1e-3, beta=1e-8):
        self.n_cmp = n_components
        self.gamma = gamma
        self.beta = beta

    @property
    def gamma(self):
        return self._gamma
   
    @property
    def beta(self):
        return self._beta

    @gamma.setter
    def gamma(self, value):
        if value <= 1 and value >= 1e-8:
            self._gamma = value
        else:
            raise ValueError('gamma value should range from 1e-8 to 1')

    @beta.setter
    def beta(self, value):
        if (value <= 1e-5) and (value >= 1e-12):
            self._beta = value
        else:
            raise ValueError('beta value should range from 1e-12 to 1e-5')

    def fit(self, train_X, train_y, train_X_dy):
        pca = PCA(self.n_cmp)
        train_tX = pca.fit_transform(train_X)
        train_tX_dy = pca.transform_gradient(train_X_dy)
        
        estimator = GPR(gamma=self.gamma, beta=self.beta, kernel=rbf_kernel,
                        optimize=True,
                        params_bounds=((1e-8, 1), (1e-12, 1e-5))) 
        estimator.fit(train_tX, train_y)
        print('Finish 1st stage fitting....')
        self.gamma = estimator.gamma        
        self.beta = estimator.beta

        gd_estimator = GPR(gamma=self.gamma, beta=self.beta, kernel=rbf_kernel_2nd_gradient,
                           optimize=False)
        gd_estimator.fit(train_tX, train_tX_dy)
        print('Finish 2nd stage fitting....')

        self.transformer = pca
        self.estimator = estimator
        self.gd_estimator = gd_estimator

    def predict(self, test_X):
        test_tX = self.transformer.transform(test_X)
        pred_y = self.estimator.predict(test_tX)
        return pred_y

    def predict_gradient(self, test_X):
        test_tX = self.transformer.transform(test_X)
        n_samples, n_features = test_tX.shape
        pred_tX_dy = self.transformer.inverse_transform_gradient(self.gd_estimator.predict(test_tX).reshape((n_samples, n_features)))
        return pred_tX_dy 
