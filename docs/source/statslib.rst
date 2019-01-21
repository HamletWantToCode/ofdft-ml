Using ML routine
================

Generate sample regression data
-------------------------------

Here, we use ``make_regression`` function in scikit-learn to generate a sample linear regression dataset::

    >>> from sklearn.datasets import make_regression
    >>> n_samples = 200
    >>> n_features = 10
    >>> n_informative = 10
    >>> effective_rank = 2
    >>> X, y, coef = make_regression(n_samples, n_features, n_informative,\
                                     effective_rank=effective_rank, coef=True,\
                                     random_state=392)
    >>> coef
    array([ 43.66010703,  90.54476838,  78.66227175,  63.70688649,
        71.19644592,  80.0509927 ,  94.26345474,  49.96992829,
         3.83384143,  36.77433012])


Kernel ridge regression
-----------------------

Use kernel ridge regression with linear kernel to fit the data::

    >>> from statslib.kernel_ridge import KernelRidge
    >>> def linear_kernel(gamma, X, Y):
    ...     return X @ Y.T
    >>> def linear_kernel_gradient(gamma, X, Y):
    ...     return np.repeat(Y.T[np.newaxis, :, :], X.shape[0], axis=0)
    >>> train_X, train_y = X[:100], y[:100]
    >>> test_X, test_y = X[100:], y[100:]
    >>> model = KernelRidge(gamma=None, C=1e-10, kernel=linear_kernel)
    >>> model.fit(train_X, train_y)
    >>> pred_y = model.predict(test_X)
    >>> coef_pred = np.sum(model.coef_[:, np.newaxis]*train_X, axis=0)
    >>> coef_pred
    array([ 43.66009507,  90.54476359,  78.6622582 ,  63.70687747,
        71.19643854,  80.05098448,  94.2634407 ,  49.96992239,
         3.83384209,  36.77432529])
    >>> pred_dy = model.predict_gradient(test_X)
    array([[ 43.66009507,  90.54476359,  78.6622582 ,  63.70687747,
         71.19643854,  80.05098448,  94.2634407 ,  49.96992239,
          3.83384209,  36.77432529],
       [ 43.66009507,  90.54476359,  78.6622582 ,  63.70687747,
         71.19643854,  80.05098448,  94.2634407 ,  49.96992239,
          3.83384209,  36.77432529], ...])
    >>> # mean square error
    >>> np.mean((pred_y - test_y)**2)
    5.6577688447796112e-13


Principal component analysis
----------------------------

PCA will help us eliminate the unimportant dimension, reduce the curse of dimensionality::

    >>> from statslib.pca import PrincipalComponentAnalysis as PCA
    >>> pca = PCA(n_components=2)
    >>> X_t = pca.fit_transform(train_X)


Grid search for optimal hyperparameter
--------------------------------------

Hyperparameter grid search can be done using ``grid_search.py`` module::

    >>> from statslib.grid_search import MyGridSearchCV
    >>> from statslib.kernel_ridge import KernelRidge
    >>> model = KernelRidge()
    >>> param_dict = {'model__C': [1e-10, 1e-5, 1e-3],
    ...               'model__gamma': [1e-2, 0.1]}
    >>> gs = MyGridSearchCV(model, param_dict, scoring='mse')
    >>> gs.fit(train_X, train_y)
    >>> gs.best_params_


