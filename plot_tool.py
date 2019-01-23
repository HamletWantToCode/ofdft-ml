# plot data

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec 

def plot_distance(distance_data, out_dir='.', **hist_params):
    fig = plt.figure()
    ax = fig.gca()
    n, bins_edge, patches = ax.hist(distance_data, density=True, **hist_params)
    bin_width = bins_edge[1] - bins_edge[0]

    def y_ticker_func(y, position):
        return np.round(y*bin_width, 2)

    ax.set_xlabel('Euclidean distance')
    ax.yaxis.set_major_formatter(FuncFormatter(y_ticker_func))
    ax.set_ylabel('fraction')
    plt.savefig('/'.join([out_dir, 'pairwise_distance.png']))

def plot_principal_components(data, out_dir='.', scatter_params={}, hist_params={}):
    fig = plt.figure(figsize=(10, 5))
    original_gs = GridSpec(1, 2, figure=fig)
    ## scatter plot
    gs_0 = GridSpecFromSubplotSpec(1, 1, original_gs[0])
    ax_0 = fig.add_subplot(gs_0[0])
    ax_0.scatter(data[:, 0], data[:, 1], **scatter_params)
    ax_0.set_xlabel('principal #1')
    ax_0.set_ylabel('principal #2')

    ## hist plot
    hist_p0, bins_p0 = np.histogram(data[:, 0], **hist_params)
    x_min, x_max = np.amin(data[:, 0]), np.amax(data[:, 0])
    gs_1 = GridSpecFromSubplotSpec(4, 2, original_gs[1], wspace=0.1)
    axes = [fig.add_subplot(gs_1[i, j]) for i in [0, 1, 2, 3] for j in [0, 1]]
    for i in range(8):
        n, bins_edge, patches = axes[i].hist(data[:, i], bins=bins_p0, density=False)
        y_max = np.amax(n)
        axes[i].xaxis.set_major_locator(FixedLocator([x_min, x_max]))
        axes[i].xaxis.set_major_formatter(NullFormatter())
        axes[i].yaxis.set_major_formatter(NullFormatter())
        axes[i].set_xlim([x_min-0.5, x_max+0.5])
        axes[i].set_ylim([0, y_max])
    axes[6].xaxis.set_major_formatter(FixedFormatter(['%.1f' %(x_min), '%.1f' %(x_max)]))
    axes[7].xaxis.set_major_formatter(FixedFormatter(['%.1f' %(x_min), '%.1f' %(x_max)]))
     
    fig.text(0.7, 0.04, 'range')
    fig.text(0.51, 0.5, 'fraction', va='center', rotation='vertical')
    plt.savefig('/'.join([out_dir, 'principal_components.png']))

def plot_real_grid_data(data_x, out_dir='.'):
    n_samples, n_points = data_x.shape
    X = np.linspace(0, 1, n_points)
    data_x_max = np.amax(data_x, axis=0)
    data_x_min = np.amin(data_x, axis=0)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(X, data_x[n_samples//2], 'k')
    ax.fill_between(X, data_x_min, data_x_max, alpha=0.7, edgecolor=None, facecolor='silver')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\rho(x)$')
    plt.savefig('/'.join([out_dir, 'density.png']))

def plot_prediction(Ek_true, Ek_pred, densx_true, densx_pred, densx_init, out_dir='.'):
    fig = plt.figure(figsize=(9, 4))
    gd = GridSpec(1, 2, figure=fig)
    ## Ek plot
    ax1 = fig.add_subplot(gd[0])
    mse_Ek = np.mean((Ek_pred-Ek_true)**2)
    ax1.plot(Ek_true, Ek_pred, 'bo', label='predict')
    ax1.plot(Ek_true, Ek_true, 'r', label='true')
    ax1.set_xlabel('Ek test')
    ax1.set_ylabel('Ek predict')
    pos_x, pos_y = 0.65*Ek_true.max() + 0.35*Ek_true.min(), 0.85*Ek_pred.min() + 0.15*Ek_pred.max()
    ax1.text(pos_x, pos_y, s='mse=%.2E' %(mse_Ek))
    ax1.legend()
    ## density plot
    n_points = densx_true.shape[0]
    X = np.linspace(0, 1, n_points)
    ax2 = fig.add_subplot(gd[1])
    ax2.plot(X, densx_pred, 'b--', label='predict')
    ax2.plot(X, densx_true, 'r', label='true', alpha=0.7)
    ax2.plot(X, densx_init, 'g--', label='initial')
    ax2.set_xlabel('x')
    ax2.set_ylabel(r'$\rho(x)$')
    ax2.legend()

    plt.savefig('/'.join([out_dir, 'predict_results.png']))


