import sys
import pickle
import getopt
import numpy as np 
import matplotlib.pyplot as plt 

def main(argv):
    def usage():
        print("""
              usage:
              -h: help
              --f_dens: density in real space
              --f_Vx: potential in real space
              """)
        return 100
    try:
        opt_vals, _ = getopt.getopt(argv, 'h', ['help', 'f_dens=', 'f_Vx='])
    except getopt.GetoptError as err:
        print(err)
        return usage()
    for opt, val in opt_vals:
        if opt in ['-h', '--help']:
            return usage()
        elif opt == '--f_dens':
            dens_fname = val
            with open(dens_fname, 'rb') as f:
                data = pickle.load(f)
            densx = data[:, 1:]
            n_samples, n_points = densx.shape
            ii = np.random.randint(0, n_samples, 1)[0]
            X = np.linspace(0, 1, n_points)
            fig1 = plt.figure(1)
            ax1 = fig1.add_subplot(111)
            ax1.fill_between(X, np.amin(densx, axis=0), np.amax(densx, axis=0), facecolor='silver')
            ax1.plot(X, densx[ii], 'k')
            ax1.set_xlabel('x')
            ax1.set_ylabel(r'$\rho(x)$')
            fig1.savefig('density_distribution.png')
        elif opt == '--f_Vx':
            Vx_fname = val
            with open(Vx_fname, 'rb') as f1:
                p_data = pickle.load(f1)
            Vx = p_data[:, 1:]
            fig2 = plt.figure(2)
            ax2 = fig2.add_subplot(111)
            ax2.fill_between(X, np.amin(Vx, axis=0), np.amax(Vx, axis=0), facecolor='silver')
            ax2.plot(X, Vx[ii], 'k')
            ax2.set_xlabel('x')
            ax2.set_ylabel('V(x)')
            fig2.savefig('Vx_distribution.png')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))