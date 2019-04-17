import numpy as np 
import warnings
from ofdft_ml.statslib.pca import PrincipalComponentAnalysis as PCA
from ofdft_ml.statslib.GaussProcess import GaussProcessRegressor as GPR
from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_2nd_gradient, approximate_rbf_kernel_hessian 

__all__ = ['GPGD_model']

class GPGD_model(object):
    def __init__(self, n_components=2, gamma=1e-3, beta=1e-8, approximate_hessian=False):
        self.n_cmp = n_components
        self.gamma = gamma
        self.beta = beta
        self.approximate_hessian = approximate_hessian

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

        if self.approximate_hessian:
            warnings.warn('Use approximation to hessian matrix of rbf kernel !')
            gd_estimator = []
            for i in range(self.n_cmp):
                _gd_estimator = GPR(
                    gamma=self.gamma,
                    beta=self.beta,
                    kernel=approximate_rbf_kernel_hessian,
                    optimize=False)
                _gd_estimator.fit(train_tX, train_tX_dy[:, i])
                gd_estimator.append(_gd_estimator)
        else:
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
        if self.approximate_hessian:
            pred_tX_dy = []
            for i in range(self.n_cmp):
                predictor = self.gd_estimator[i]
                pred_tX_dyi = predictor.predict(test_tX)
                pred_tX_dy.append(pred_tX_dyi)
            pred_tX_dy = np.transpose(np.array(pred_tX_dy))
        else:
            pred_tX_dy = self.gd_estimator.predict(test_tX).reshape((n_samples, n_features))
        pred_X_dy = self.transformer.inverse_transform_gradient(pred_tX_dy)
        return pred_X_dy 
