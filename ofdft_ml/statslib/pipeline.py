from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.utils.metaestimators import if_delegate_has_method

__all__ = ['NewPipeline']

class NewPipeline(Pipeline):
    """
    Pipline is used to encapsulate transformers and estimator

    :param steps: list of tuples, e.g. [('transform', pca), ('estimator', krr)]

    :param memory: None, str or object with the joblib.Memory interface, optional, used to cache the fitted transformers of the pipeline.

    Examples::

    >>> from statslib.pipline import MyPipe
    >>> from statslib.pca import PrincipalComponentAnalysis
    >>> from statslib.kernel_ridge import KernelRidge
    >>> import numpy as np
    >>> pca = PrincipalComponentAnalysis()
    >>> krr = KernelRidge()
    >>> pipe = Mypipe([('PCA', pca), ('KRR', krr)])
    >>> X = np.random.rand(10, 5)
    >>> y = np.random.rand(10)
    >>> pipe.fit(X, y)
    """
    def __init__(self, steps, memory=None):
        self.steps = steps
        self._validate_steps()

    # def _fit(self, X, y=None, dy=None):
    #     Xt, dyt = X, dy
    #     for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
    #         if transformer is None:
    #             pass
    #         else:
    #             # Fit the current transfomer
    #             Xt, fitted_transformer = _fit_transform_one(transformer, None, Xt, y)
    #             # Fit the gradient
    #             if dy is not None:
    #                 dyt = fitted_transformer.transform_gradient(dy)
    #             # Replace the transformer of the step with the fitted
    #             # transformer. 
    #             self.steps[step_idx] = (name, fitted_transformer)
    #     return Xt

    def fit(self, X, y=None):
        """
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        :param X: iterable, training data. Must fulfill input requirements of first step of the pipeline.

        :param y: iterable, default=None, training targets. Must fulfill label requirements for all steps of the pipeline.

        :param dy: iterable, default=None, training function gradient

        :return self: Pipeline, this estimator
        """
        name, transformer = self.steps[0]
        Xt, fitted_transformer = _fit_transform_one(transformer, None, X, y)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y)
        return self

    def fit_gradient(self, X, dy=None):
        name, transfomer = self.steps[0]
        Xt, fitted_transformer = _fit_transform_one(transfomer, None, X, dy)
        dyt = fitted_transformer.transform_gradient(dy)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, dyt)
        return self

    # @if_delegate_has_method(delegate='_final_estimator')
    # def predict_gradient(self, X):
    #     Xt = X
    #     for name, transformer in self.steps[:-1]:
    #         if transformer is not None:
    #             Xt = transformer.transform(Xt)
    #     dyt_predict = self.steps[-1][-1].predict_gradient(Xt)
    #     # !!! for real space grid, we have to multiply the grid number
    #     return (X.shape[1]-1)*dyt_predict

    # def fit_transform(self, X, y=None):
    #     last_step = self._final_estimator
    #     Xt, fit_params = self._fit(X, y)
    #     if hasattr(last_step, 'fit_transform'):
    #         return last_step.fit_transform(Xt, y)
    #     elif last_step is None:
    #         return Xt
    #     else:
    #         return last_step.fit(Xt, y).transform(Xt)
