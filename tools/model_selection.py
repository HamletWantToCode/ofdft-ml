#!/usr/bin/env python

import sys
import getopt
import numpy as np
import pickle
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mpi4py import MPI

from ofdft_ml.statslib.kernel_ridge import KernelRidge
from ofdft_ml.statslib.pca import PrincipalComponentAnalysis
from ofdft_ml.statslib.new_grid_search import NewGridSearchCV
from ofdft_ml.statslib.new_pipeline import NewPipeline
from ofdft_ml.statslib.new_scorer import make_scorer
from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_gradient

def main(argv, R=np.random.RandomState(32892)):
    # MPI setup
    comm = MPI.COMM_WORLD
    SIZE = comm.Get_size()
    ID = comm.Get_rank()
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
    pipe = NewPipeline([('reduce_dim', PrincipalComponentAnalysis()),\
                   ('regressor', KernelRidge(kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient))])
    # Distribute parameters to each process
    all_params = product(N_COMPONENTS, list(np.logspace(LOW_C, HIGH_C, N_GRIDS)),\
                         list(np.logspace(LOW_GAMMA, HIGH_GAMMA, N_GRIDS)))
    all_params_list = list(all_params)
    param_grid = []
    for n_components_, c_, gamma_ in all_params_list[ID::SIZE]:
        param_grid.append(
                    {
                    'reduce_dim__n_components': [n_components_],
                    'regressor__C': [c_],        
                    'regressor__gamma': [gamma_] 
                    }
        )
    grid_search = NewGridSearchCV(pipe, param_grid, cv=n_CV, scoring=neg_mean_squared_error_scorer)
    grid_search.fit(densx_train, Ek_train, dEkx_train)
    best_params_on_node = grid_search.best_params_
    best_estimator_on_node = grid_search.best_estimator_
    best_score_on_node = abs(grid_search.best_score_)
    collect_scores = np.zeros((SIZE, 2))
    # Gather data from each process
    comm.Allgather(np.array([best_score_on_node, ID]), collect_scores)
    best_rank = int(collect_scores[np.argmin(collect_scores[:, 0]), 1])
    if ID == best_rank:
        print(print('best parameters:\n', best_params_on_node, '\n'))
        print('test score (mse):', best_score_on_node)
        with open('demo_best_estimator', 'wb') as f2:
            pickle.dump(best_estimator_on_node, f2)
        with open('demo_train_data', 'wb') as f3:
            train_data = np.c_[Ek_train.reshape((-1, 1)), densx_train, dEkx_train]
            pickle.dump(train_data, f3)
        with open('demo_test_data', 'wb') as f4:
            test_data = np.c_[Ek_test.reshape((-1, 1)), densx_test, dEkx_test]
            pickle.dump(test_data, f4)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))