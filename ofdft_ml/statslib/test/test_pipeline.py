# test pipeline

from numpy.testing import assert_array_almost_equal
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from ofdft_ml.statslib.pipeline import NewPipeline

def test_pipline_common():
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.kernel_ridge import KernelRidge
    # make dataset
    X, y = make_regression(n_features=10, random_state=0)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=3)
    # build model
    pca = PCA()
    krr = KernelRidge()
    sklearn = Pipeline([('PCA', pca), ('regression', krr)])
    my = NewPipeline([('PCA', pca), ('regression', krr)])
    sklearn.fit(train_X, train_y)
    my.fit(train_X, train_y)
    # check prediction
    sklearn_pred_y = sklearn.predict(test_X)
    my_pred_y = my.predict(test_X)
    assert_array_almost_equal(sklearn_pred_y, my_pred_y, 7)

#def test_pipeline_gradient():
#    import pickle
#    from ofdft_ml.statslib.pca import PrincipalComponentAnalysis as PCA
#    from ofdft_ml.statslib.old.kernel_ridge import KernelRidge
#    from ofdft_ml.statslib.utils import rbf_kernel_gradient
#    from ofdft_ml.statslib.old import MyPipe
#
#    with open('ofdft_ml/statslib/test/test_densx_Ek', 'rb') as f:
#        data = pickle.load(f)
#    X, y = data[:, 1:], data[:, 0]
#    with open('ofdft_ml/statslib/test/test_Vx_mu', 'rb') as f:
#        data = pickle.load(f)
#    dy = -data[:, 1:]
#    train_X, test_X, train_y, test_y, train_dy, test_dy = \
#        train_test_split(X, y, dy, test_size=0.4, random_state=3)
#    # build model
#    pca = PCA()
#    krr = KernelRidge(kernel_gd=rbf_kernel_gradient)
#    old = MyPipe([('PCA', pca), ('regressor', krr)])
#    new = NewPipeline([('PCA', pca), ('regressor', krr)])
#    old.fit(train_X, train_y, train_dy)
#    new.fit(train_X, train_y, train_dy)
#    # check prediction
#    old_pred_y = old.predict(test_X)
#    old_pred_dy = old.predict_gradient(test_X)
#    new_pred_y = new.predict(test_X)
#    new_pred_dy = new.predict_gradient(test_X)
#    assert_array_almost_equal(old_pred_y, new_pred_y, 7)
#    assert_array_almost_equal(old_pred_dy, new_pred_dy, 7)
