#!/usr/bin/env python

import pickle
import sys
import getopt
import numpy as np 
from sklearn.model_selection import train_test_split

from ofdft_ml.statslib.pipeline import NewPipeline
from ofdft_ml.statslib.pca import PrincipalComponentAnalysis as PCA
from ofdft_ml.statslib.GaussProcess import GaussProcessRegressor
from ofdft_ml.statslib.utils import rbf_kernel, rbf_kernel_2nd_gradient

def main(argv):
    def usage():
        print("""
              usage:
              --f_dens: input file contains density and Ek\n
              --f_grad: input file contains potential\n
              -r: test ratio\n
              --params: model parameter file for CV
              """)
        return 100
    try:
        opt_vals, _ = getopt.getopt(argv, 'hr:', ['f_dens=', 'f_grad=', 'params='])
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
                mu, X_Vx = potential[:, :1], potential[:, 1:]
                X_dEk = mu - X_Vx
        elif opt == '-r': test_ratio = float(val)
        elif opt == '--params':
            with open(val, 'r') as f:
                for line in f:
                    ii = line.index('=')
                    if line[:ii] == 'n_components':
                        N_COMPONENTS = int(line[(ii+1):])
                        # N_COMPONENTS = [int(v) for v in nums]
                    elif line[:ii] == 'gamma':
                        gamma = float(line[(ii+1):])
                    elif line[:ii] == 'beta':
                        beta = float(line[(ii+1):])
                    elif line[:ii] == 'params_bounds':
                        bounds = line[(ii+1):].split(':')
                        bounds = ((float(bounds[0]), float(bounds[1])), (float(bounds[2]), float(bounds[3])))

    train_X_density, test_X_density, train_Ek, test_Ek,\
                               train_X_dEk, test_X_dEk = train_test_split(X_density, Ek, X_dEk, test_size=test_ratio, random_state=5)
    
    # fitting the function
    pipe_func = NewPipeline([('reduce_dim', PCA(N_COMPONENTS)),
                             ('regressor', GaussProcessRegressor(gamma, beta, kernel=rbf_kernel,
                                                                 optimize=True,
                                                                 params_bounds=bounds))])
    pipe_func.fit(train_X_density, train_Ek)
    best_gamma = pipe_func.named_steps['regressor'].gamma
    best_beta = pipe_func.named_steps['regressor'].beta

    # fitting the gradient
    pipe_gradient = NewPipeline([('reduce_dim', PCA(N_COMPONENTS)),
                                 ('regressor', GaussProcessRegressor(gamma=best_gamma, beta=best_beta,
                                                                     kernel=rbf_kernel_2nd_gradient,
                                                                     optimize=False))])
    pipe_gradient.fit_gradient(train_X_density, train_X_dEk)

    print("""
          The optimal parameters after training are:
            PCA components: %d\n
            GP gamma:       %.5f\n
            GP beta:        %.5E\n
          """ %(N_COMPONENTS, best_gamma, best_beta))

    with open('best_estimator', 'wb') as f_out:
        pickle.dump(pipe_func, f_out)
    with open('best_gd_estimator', 'wb') as fgd_out:
        pickle.dump(pipe_gradient, fgd_out)
    with open('train_data', 'wb') as f_train:
        train_data = np.c_[train_Ek.reshape((-1, 1)), train_X_density, train_X_dEk]
        pickle.dump(train_data, f_train)
    with open('test_data', 'wb') as f_test:
        test_data = np.c_[test_Ek.reshape((-1, 1)), test_X_density, test_X_dEk]
        pickle.dump(test_data, f_test)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))