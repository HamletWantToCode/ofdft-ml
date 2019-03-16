#!/usr/bin/env python

import sys
import getopt
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import *

def density_projection(dens_x):
    from ofdft_ml.statslib.pca import PrincipalComponentAnalysis
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec 
    
    pca = PrincipalComponentAnalysis(8)
    dens_t = pca.fit_transform(dens_x)
    tr_mat = pca.tr_mat_
    # plots
    ## density data distribution
    dens_fig = plt.figure(0, figsize=(10, 5))
    dens_gs = GridSpec(1, 2)
    ### scatter plot
    gs_0 = GridSpecFromSubplotSpec(1, 1, dens_gs[0])
    ax_0 = dens_fig.add_subplot(gs_0[0])
    ax_0.scatter(dens_t[:, 0], dens_t[:, 1])
    ax_0.set_xlabel('principal #1')
    ax_0.set_ylabel('principal #2')
    ### histogram plot
    _, bins_p0 = np.histogram(dens_t[:, 0], bins=20, density=False)
    x_min, x_max = np.amin(dens_t[:, 0]), np.amax(dens_t[:, 0])
    gs_1 = GridSpecFromSubplotSpec(4, 2, dens_gs[1], wspace=0.1)
    dens_axes = [dens_fig.add_subplot(gs_1[i, j]) for i in [0, 1, 2, 3] for j in [0, 1]]
    for i in range(8):
        n, bins_edge, patches = dens_axes[i].hist(dens_t[:, i], bins=bins_p0)
        y_max = np.amax(n)
        dens_axes[i].xaxis.set_major_locator(FixedLocator([x_min, x_max]))
        dens_axes[i].xaxis.set_major_formatter(NullFormatter())
        dens_axes[i].yaxis.set_major_formatter(NullFormatter())
        dens_axes[i].set_xlim([x_min-0.5, x_max+0.5])
        dens_axes[i].set_ylim([0, y_max])
    dens_axes[6].xaxis.set_major_formatter(FixedFormatter(['%.1f' %(x_min), '%.1f' %(x_max)]))
    dens_axes[7].xaxis.set_major_formatter(FixedFormatter(['%.1f' %(x_min), '%.1f' %(x_max)]))
    dens_fig.text(0.7, 0.04, 'range')
    dens_fig.text(0.51, 0.5, 'fraction', va='center', rotation='vertical')
    dens_fig.savefig('principal_components.png')
    ## transfer matrix plot
    X = np.linspace(0, 1, dens_x.shape[1])
    mat_fig = plt.figure(1, figsize=(5, 5))
    mat_gs = GridSpec(2, 2, wspace=0.2)
    mat_axes = [mat_fig.add_subplot(mat_gs[i, j]) for i in [0, 1] for j in [0, 1]]
    for i in range(4):
        mat_axes[i].plot(X, tr_mat[:, i], label='#%s' %(i))
        vec_min, vec_max = np.amin(tr_mat[:, i]), np.amax(tr_mat[:, i])
        mat_axes[i].legend()
        mat_axes[i].xaxis.set_major_locator(FixedLocator([0, 1]))
        mat_axes[i].yaxis.set_major_locator(FixedLocator([vec_min, vec_max]))
        mat_axes[i].xaxis.set_major_formatter(NullFormatter())
        mat_axes[i].yaxis.set_major_formatter(FixedFormatter(['%.2f' %(vec_min), '%.2f' %(vec_max)]))
    mat_axes[2].xaxis.set_major_formatter(FixedFormatter(['0', '1']))
    mat_axes[3].xaxis.set_major_formatter(FixedFormatter(['0', '1']))
    mat_fig.savefig('transfer_matrix.png')
    return 0

def predict_results(Ek, Ek_, dens, dens_, dens_true, grad, grad_):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec 

    fig = plt.figure(figsize=(12, 4))
    gd = GridSpec(1, 3)
    ## Ek plot
    ax1 = fig.add_subplot(gd[0])
    mse_Ek = np.mean((Ek-Ek_)**2)
    ax1.plot(Ek, Ek_, 'bo', label='predict')
    ax1.plot(Ek, Ek, 'r', label='true')
    ax1.set_xlabel('Ek test')
    ax1.set_ylabel('Ek predict')
    pos_x, pos_y = 0.65*Ek.max() + 0.35*Ek.min(), 0.85*Ek_.min() + 0.15*Ek_.max()
    ax1.text(pos_x, pos_y, s='mse=%.2E' %(mse_Ek))
    ax1.set_title('Ek')
    ax1.legend()
    ## density plot
    n_space = len(dens)
    X = np.linspace(0, 1, n_space)
    ax2 = fig.add_subplot(gd[1])
    ax2.plot(X, dens_, 'b--', label='predict')
    ax2.plot(X, dens_true, 'r', label='true', alpha=0.7)
    ax2.plot(X, dens, 'g--', label='initial')
    ax2.set_xlabel('x')
    ax2.set_ylabel(r'$\rho(x)$')
    ax2.set_title('density')
    ax2.legend()
    ## grad plot
    ax3 = fig.add_subplot(gd[2])
    ax3.plot(X, grad, 'r', label='true')
    ax3.plot(X, grad_, 'b--', label='predict')
    ax3.set_xlabel('x')
    ax3.set_ylabel(r'$\frac{\delta T}{\delta n(x)}$')
    ax3.set_title('gradient')
    ax3.legend()
    fig.savefig('predict_results.png')
    return 0

def main(argv):
    import pickle
    from ofdft_ml.quantum.EL_solver import EulerSolver

    def usage():
        print("""
                usage:
                -f proc: data preprocessing show the data distribution\n
                -f --params pred: doing prediction and plot results\n
              """)
        return 100
    try:
        opt_vals, args = getopt.getopt(argv, 'hf:', ['help', 'file=', 'params='])
    except getopt.GetoptError as err:
        print(err)
        return usage()
    if opt_vals[0][0] in ['-h', '--help']: return usage()
    if args[0] == 'proc':
        opt, val = opt_vals[0]
        if opt in ['-f', '--file']:
            with open(val, 'rb') as f:
                data = pickle.load(f)
            dens_x = data[:, 1:]
            density_projection(dens_x)
        else:
            return usage()
    elif args[0] == 'pred':
        for (opt, val) in opt_vals:
            if opt in ['-f', '--file']:
                fnames = val
                train_fname, test_fname, estimator_fname, gd_estimator_fname = fnames.split(':')
                with open(train_fname, 'rb') as f:
                    train_data = pickle.load(f)
                n_space = train_data.shape[1] - 1
                train_densx, train_Ek, train_gradx = train_data[:, 1:1+n_space//2], train_data[:, 0], train_data[:, 1+n_space//2:]
                with open(test_fname, 'rb') as f1:
                    test_data = pickle.load(f1)
                test_densx, test_Ek, test_gradx = test_data[:, 1:1+n_space//2], test_data[:, 0], test_data[:, 1+n_space//2:]
                with open(estimator_fname, 'rb') as f2:
                    estimator = pickle.load(f2)
                with open(gd_estimator_fname, 'rb') as f3:
                    gd_estimator = pickle.load(f3)
            elif opt == '--params':
                param_fname = val 
                with open(param_fname, 'r') as ft:
                    for line in ft:
                        ii = line.index('=')
                        if line[:ii] == 'mu': mu = float(line[(ii+1):])
                        elif line[:ii] == 'n': n_electron = int(line[(ii+1):])
                        elif line[:ii] == 'step': step = float(line[(ii+1):])
                        elif line[:ii] == 'tol': tol = float(line[(ii+1):])
            else:
                return usage()
        # Ek prediction
        pred_Ek = estimator.predict(test_densx)
        # density optimization
        i = np.random.randint(0, len(train_data))
        j = np.random.randint(0, len(test_data))
        dens_init = train_densx[i]
        dens_targ, gradx_targ = test_densx[j], test_gradx[j]
        Vx_targ = -gradx_targ
        pred_gradx_t = gd_estimator.predict(dens_targ[np.newaxis, :])[np.newaxis, :]
        pred_gradx = gd_estimator.named_steps['reduce_dim'].inverse_transform_gradient(pred_gradx_t)[0]
        optimizer = EulerSolver(estimator, gd_estimator)
        dens_predict = optimizer.run(dens_init[np.newaxis, :], Vx_targ, mu,\
                                      n_electron, step, tol, verbose=True)
        predict_results(test_Ek, pred_Ek, dens_init, dens_predict, dens_targ, gradx_targ, pred_gradx)
        
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
        






