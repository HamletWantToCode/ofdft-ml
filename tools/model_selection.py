#!/usr/bin/env python

import sys
import getopt
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ofdft_ml.statslib.kernel_ridge import KernelRidge
from ofdft_ml.statslib.pca import PrincipalComponentAnalysis
from ofdft_ml.statslib.grid_search import MyGridSearchCV
from ofdft_ml.statslib.pipeline import MyPipe
from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_gradient
from ofdft_ml.statslib.scorer import make_scorer

def main(argv, R=np.random.RandomState(32892)):
    # parsing parameter
    def usage():
        print("""
              usage:
              --f_dens: input file contains density and Ek\n
              --f_grad: input file contains potential\n
              -r: test ratio\n
              -n: number of folds for CV\n
              --params: model parameter file for CV
              """)
        return 100
    try:
        opt_vals, _ = getopt.getopt(argv, 'hr:n:', ['f_dens=', 'f_grad=', 'params='])
    except getopt.GetoptError as err:
        print(err)
        return usage()
    for (opt, val) in opt_vals:
        if opt in ['-h', '--help']: return usage()
        elif opt == '--f_dens':
            with open(val, 'rb') as f:
                data = pickle.load(f)
                densx, Ek = data[:, 1:], data[:, 0]
        elif opt == '--f_grad':
            with open(val, 'rb') as f1:
                potential = pickle.load(f1)
                dEkx = -potential[:, 1:]
        elif opt == '-r': test_ratio = float(val)
        elif opt == '-n': n_CV = int(val)
        elif opt == '--params':
            with open(val, 'r') as f:
                for line in f:
                    ii = line.index('=')
                    if line[:ii] == 'n_components':
                        nums = line[(ii+1):].split(':')
                        N_COMPONENTS = [int(v) for v in nums]
                    elif line[:ii] == 'C':
                        low, up = line[(ii+1):].split(':')
                        LOW_C, HIGH_C = float(low), float(up)
                    elif line[:ii] == 'gamma':
                        low, up = line[(ii+1):].split(':')
                        LOW_GAMMA, HIGH_GAMMA = float(low), float(up)
                    elif line[:ii] == 'ngrid': N_GRIDS = int(line[(ii+1):])
    densx_train, densx_test, Ek_train, Ek_test, dEkx_train, dEkx_test = train_test_split(densx,\
                                                      Ek, dEkx, test_size=test_ratio, random_state=R)
    neg_mean_squared_error_scorer = make_scorer(mean_squared_error)
    pipe = MyPipe([('reduce_dim', PrincipalComponentAnalysis()),\
                   ('regressor', KernelRidge(kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient))])
    param_grid = [
                {
                'reduce_dim__n_components': N_COMPONENTS,
                'regressor__C': np.logspace(LOW_C, HIGH_C, N_GRIDS),        
                'regressor__gamma': np.logspace(LOW_GAMMA, HIGH_GAMMA, N_GRIDS)
                }
                ]
    grid_search = MyGridSearchCV(pipe, param_grid, cv=n_CV, scoring=neg_mean_squared_error_scorer)
    grid_search.fit(densx_train, Ek_train, dEkx_train)
    print('best parameters:\n', grid_search.best_params_, '\n')
    test_score = grid_search.cv_results_['mean_test_score']
    best_score_index = grid_search.best_index_
    print('test score (mse):', -test_score[best_score_index])

    best_estimator = grid_search.best_estimator_

    with open('demo_best_estimator', 'wb') as f2:
        pickle.dump(best_estimator, f2)
    with open('demo_train_data', 'wb') as f3:
        train_data = np.c_[Ek_train.reshape((-1, 1)), densx_train, dEkx_train]
        pickle.dump(train_data, f3)
    with open('demo_test_data', 'wb') as f4:
        test_data = np.c_[Ek_test.reshape((-1, 1)), densx_test, dEkx_test]
        pickle.dump(test_data, f4)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))