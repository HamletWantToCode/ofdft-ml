#!/usr/bin/env python

import pickle
import sys
import getopt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from ofdft_ml.statslib.pipeline import NewPipeline
from ofdft_ml.statslib.pca import PrincipalComponentAnalysis as PCA
from ofdft_ml.statslib.GaussProcess import GaussProcessRegressor
from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_gradient

def main(argv):
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
                X_density, Ek = data[:, 1:], data[:, 0]
        elif opt == '--f_grad':
            with open(val, 'rb') as f1:
                potential = pickle.load(f1)
                X_dEk = -potential[:, 1:]
        elif opt == '-r': test_ratio = float(val)
        elif opt == '-n': n_CV = int(val)
        elif opt == '--params':
            with open(val, 'r') as f:
                for line in f:
                    ii = line.index('=')
                    if line[:ii] == 'n_components':
                        nums = line[(ii+1):].split(':')
                        N_COMPONENTS = [int(v) for v in nums]
                    elif line[:ii] == 'gamma':
                        gamma = float(line[(ii+1):])
                    elif line[:ii] == 'beta':
                        beta = float(line[(ii+1):])
                    elif line[:ii] == 'params_bounds':
                        bounds = line[(ii+1):].split(':')
                        bounds = ((float(bounds[0]), float(bounds[1])), (float(bounds[2]), float(bounds[3])))

    train_X_density, test_X_density, train_Ek, test_Ek,\
                               train_X_dEk, test_X_dEk = train_test_split(X_density, Ek, X_dEk, test_size=test_ratio, random_state=5)

    GPR = GaussProcessRegressor(gamma, beta, kernel=rbf_kernel,
                                kernel_gd=rbf_kernel_gradient, optimize=True,
                                params_bounds=bounds)
    pipe = NewPipeline([('reduce_dim', PCA()), ('regressor', GPR)])
    gCV = GridSearchCV(pipe, param_grid={'reduce_dim__n_components': N_COMPONENTS}, 
                       cv=n_CV, scoring='neg_mean_squared_error')

    gCV.fit(train_X_density, train_Ek)
    best_estimator = gCV.best_estimator_
    n_components_ = best_estimator.named_steps['reduce_dim'].n_components
    gamma_ = best_estimator.named_steps['regressor'].gamma
    beta_ = best_estimator.named_steps['regressor'].beta
    print("""
          The optimal parameters after training are:
            PCA components: %d\n
            GP gamma:       %.5f\n
            GP beta:        %.5E\n
          """ %(n_components_, gamma_, beta_))

    with open('demo_best_estimator', 'wb') as f_out:
        pickle.dump(best_estimator, f_out)
    with open('demo_train_data', 'wb') as f_train:
        train_data = np.c_[train_Ek.reshape((-1, 1)), train_X_density, train_X_dEk]
        pickle.dump(train_data, f_train)
    with open('demo_test_data', 'wb') as f_test:
        test_data = np.c_[test_Ek.reshape((-1, 1)), test_X_density, test_X_dEk]
        pickle.dump(test_data, f_test)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))