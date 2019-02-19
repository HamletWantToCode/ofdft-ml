import sys
import pickle
import getopt
import numpy as np 
import matplotlib.pyplot as plt 

def main(argv):
    opt_vals, _ = getopt.getopt(argv, '', ['f_dens=', 'f_Vx='])
    for opt, val in opt_vals:
        if opt == '--f_dens':
            dens_fname = val
        elif opt == '--f_Vx':
            Vx_fname = val
    with open(dens_fname, 'rb') as f:
        data = pickle.load(f)
    densx = data[:, 1:]
    with open(Vx_fname, 'rb') as f1:
        p_data = pickle.load(f1)
    Vx = p_data[:, 1:]
    assert densx.shape == Vx.shape, print('data shape mismatch !')
    n_samples, n_points = densx.shape
    ii = np.random.randint(0, n_samples, 1)
    X = np.linspace(0, 1, n_points)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.fill_between(X, np.amin(densx, axis=0), np.amax(densx, axis=0), facecolor='silver')
    ax1.plot(X, densx[ii], 'k')
    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$\rho(x)$')
    ax2.fill_between(X, np.amin(Vx, axis=0), np.amax(Vx, axis=0), facecolor='silver')
    ax2.plot(X, Vx[ii], 'k')
    ax2.set_xlabel('x')
    ax2.set_ylabel('V(x)')
    fig.savefig('data_distribution.png')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))