# extended pipline

from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.base import clone
from sklearn.externals import six
from sklearn.utils.validation import check_memory
from sklearn.utils.metaestimators import if_delegate_has_method

class MyPipe(Pipeline):
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
        super().__init__(steps, memory)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_gradient(self, X, **predict_params):
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        dyt_predict = self.steps[-1][-1].predict_gradient(Xt, **predict_params)
        # !!! for real space grid, we have to multiply the grid number
        return (X.shape[1]-1)*dyt_predict

    # @if_delegate_has_method(delegate='_final_estimator')
    # def predict_hessian(self, X, **predict_params):
    #     Xt = X
    #     for name, transform in self.steps[:-1]:
    #         if transform is not None:
    #             Xt = transform.transform(Xt)
    #     ddyt_predict = self.steps[-1][-1].predict_hessian(Xt, **predict_params)
    #     return (X.shape[1]-1)*ddyt_predict

    def _fit(self, X, y=None, dy=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        dyt = dy
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer is None:
                pass
            else:
                if hasattr(memory, 'location'):
                    # joblib >= 0.12
                    if memory.location is None:
                        # we do not clone when caching is disabled to
                        # preserve backward compatibility
                        cloned_transformer = transformer
                    else:
                        cloned_transformer = clone(transformer)
                elif hasattr(memory, 'cachedir'):
                    # joblib < 0.11
                    if memory.cachedir is None:
                        # we do not clone when caching is disabled to
                        # preserve backward compatibility
                        cloned_transformer = transformer
                    else:
                        cloned_transformer = clone(transformer)
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name])
                # Fit the gradient
                if dy is not None:
                    dyt = fitted_transformer.transform_gradient(dy)
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator is None:
            return Xt, dyt, {}
        return Xt, dyt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, dy=None, **fit_params):
        """
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        :param X: iterable, training data. Must fulfill input requirements of first step of the pipeline.

        :param y: iterable, default=None, training targets. Must fulfill label requirements for all steps of the pipeline.

        :param dy: iterable, default=None, training function gradient

        :param **fit_params: dict of string -> object\
                             parameters passed to the ``fit`` method of each step, where\
                             each parameter name is prefixed such that parameter ``p`` for step\
                             ``s`` has key ``s__p``.

        :return self: Pipeline, this estimator
        """
        Xt, dyt, fit_params = self._fit(X, y, dy, **fit_params)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y, dyt, **fit_params)
        return self
