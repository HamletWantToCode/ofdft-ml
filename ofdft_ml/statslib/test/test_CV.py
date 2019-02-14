# test grid search CV

from numpy.testing import assert_almost_equal, assert_array_almost_equal
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from ofdft_ml.statslib.new_grid_search import NewGridSearchCV

def test_grid_search_common():
    from sklearn.datasets import make_regression
    from sklearn.model_selection import GridSearchCV
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import make_scorer

    # prepare database
    X, y = make_regression(n_samples=100, n_features=10, random_state=0)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=5)
    params_grid = {'alpha': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-2, 0.1, 1]}
    model = KernelRidge(kernel='rbf')
    neg_mean_square_error = make_scorer(mean_squared_error, greater_is_better=False)
    # setup model
    sklearn = GridSearchCV(model, params_grid, scoring=neg_mean_square_error)
    my = NewGridSearchCV(model, params_grid, neg_mean_square_error)
    sklearn.fit(train_X, train_y)
    my.fit(train_X, train_y)
    ## check parameters are equal
    sklearn_best_params = sklearn.best_params_
    my_best_params = my.best_params_
    for key, value in sklearn_best_params.items():
        assert_almost_equal(value, my_best_params[key], 7)
    ## check prediction is the same
    pred_y_sklearn = sklearn.predict(test_X)
    pred_y_my = my.predict(test_X)
    assert_almost_equal(pred_y_sklearn, pred_y_my, 7)
    ## check mean_test_error is the same
    score_sklearn = sklearn.cv_results_['mean_test_score']
    score_my = my.cv_results_['mean_test_score']
    assert_array_almost_equal(score_sklearn, score_my, 7)

def test_grid_search_gradient():
    import pickle
    from ofdft_ml.statslib.pca import PrincipalComponentAnalysis as pca
    from ofdft_ml.statslib.pipeline import MyPipe
    from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_gradient
    from ofdft_ml.statslib.kernel_ridge import KernelRidge
    from ofdft_ml.statslib.grid_search import MyGridSearchCV
    from ofdft_ml.statslib.new_scorer import make_scorer as new_make_scorer
    from ofdft_ml.statslib.scorer import make_scorer

    path = '/Users/hongbinren/Documents/Code/iop/ofdft-ml/ofdft_ml/statslib/test/'
    with open(path+'test_densx_Ek', 'rb') as f:
        data = pickle.load(f)
    X, y = data[:, 1:], data[:, 0]
    with open(path+'test_Vx_mu', 'rb') as f:
        data = pickle.load(f)
    dy = -data[:, 1:]
    train_X, test_X, train_y, test_y, train_dy, test_dy = \
        train_test_split(X, y, dy, test_size=0.4, random_state=3)
    new_scorer = new_make_scorer(mean_squared_error)
    old_scorer = make_scorer(mean_squared_error)
    model = MyPipe([('reduce_dim', pca()), ('regressor', KernelRidge(kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient))])
    params_grid = {'regressor__C': [1e-10, 1e-5, 1e-3], 'regressor__gamma': [1e-3, 1e-2, 0.1]}
    # setup model
    old_CV = MyGridSearchCV(model, params_grid, old_scorer, cv=3)
    new_CV = NewGridSearchCV(model, params_grid, new_scorer, cv=3)
    old_CV.fit(train_X, train_y, train_dy)
    new_CV.fit(train_X, train_y, train_dy)
    # check parameters are equal
    old_best_params = old_CV.best_params_
    new_best_params = new_CV.best_params_
    for key, value in old_best_params.items():
        assert_almost_equal(value, new_best_params[key], 7)
    # NOTE: the "scores" between new and old CV are different, due to the
    # different feature number, in old version, feature number = real grid
    # in new version, feature number = reduced dimension !!!