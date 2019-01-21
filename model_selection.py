# machine learning with cross validation

import numpy as np
import pickle
from absl import app, flags
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from statslib.kernel_ridge import KernelRidge
from statslib.pca import PrincipalComponentAnalysis
from statslib.grid_search import MyGridSearchCV
from statslib.pipeline import MyPipe
from statslib.utils import rbf_kernel, rbf_kernel_gradient
from statslib.scorer import make_scorer

FLAGS = flags.FLAGS
flags.DEFINE_list("N_COMPONENTS", [0], "list of principal component number")
flags.DEFINE_float("LOW_C", 1e-10, "lower bound for parameter C")
flags.DEFINE_float("HIGH_C", 1e-5, "upper bound for parameter C")
flags.DEFINE_float("LOW_GAMMA", 1e-5, "lower bound for parameter gamma")
flags.DEFINE_float("HIGH_GAMMA", 1e-1, "upper bound for parameter gamma")
flags.DEFINE_integer("N_POINTS", 0, "number of points in each parameter dimension")

def main(argv):
    del argv

    R = np.random.RandomState(328392)

    with open('results/demo_data', 'rb') as f:
        data = pickle.load(f)
    with open('results/demo_Vx', 'rb') as f1:
        potential = pickle.load(f1)
    densx, Ek, dEkx = data[:, 2:], data[:, 1], -potential[:, 1:]
    densx_train, densx_test, Ek_train, Ek_test, dEkx_train, dEkx_test = train_test_split(densx, Ek, dEkx, test_size=0.4, random_state=R)

    neg_mean_squared_error_scorer = make_scorer(mean_squared_error)
    pipe = MyPipe([('reduce_dim', PrincipalComponentAnalysis()), ('regressor', KernelRidge(kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient))])
    param_grid = [
                {
                'reduce_dim__n_components': FLAGS.N_COMPONENTS,
                'regressor__C': np.logspace(FLAGS.LOW_C, FLAGS.HIGH_C, N_POINTS),        
                'regressor__gamma': np.logspace(FLAGS.LOW_GAMMA, FLAGS.HIGH_GAMMA, N_POINTS)
                }
                ]
    grid_search = MyGridSearchCV(pipe, param_grid, cv=5, scoring=neg_mean_squared_error_scorer)
    grid_search.fit(densx_train, Ek_train, dEkx_train)
    print('best parameters:\n', grid_search.best_params_, '\n')
    test_score = grid_search.cv_results_['mean_test_score']
    best_score_index = grid_search.cv_results_['rank_test_score'][0]-1
    print('test score (mse):', -test_score[best_score_index])

    best_estimator = grid_search.best_estimator_

    with open('results/demo_best_estimator', 'wb') as f2:
        pickle.dump(best_estimator, f2)
    with open('results/demo_train_data', 'wb') as f3:
        train_data = np.c_[Ek_train.reshape((-1, 1)), densx_train, dEkx_train]
        pickle.dump(train_data, f3)
    with open('results/demo_test_data', 'wb') as f4:
        test_data = np.c_[Ek_test.reshape((-1, 1)), densx_test, dEkx_test]
        pickle.dump(test_data, f4)

if __name__ == '__main__':
    app.run(main)