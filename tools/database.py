#!/usr/bin/env python

import sys
import getopt
import numpy as np
from mpi4py import MPI
import pickle

def main(argv):
    # parallel setting
    comm = MPI.COMM_WORLD
    SIZE = comm.Get_size()
    ID = comm.Get_rank()
    RANDOM_SEED = ID
    # argument parsing
    def usage():
        print("""
              usage:
              -n: number of data samples in database\n
              x: use real space solver to solve atom problem\n
                  --params: n_points, n_Gauss, a, b, c, ne
              k: use k space solver to solve lattice problem\n
                  --params: n_basis, n_kpoints, n_cos, V0, Phi0, occ
              """)
        return 100 
    try:
        opt_vals, args = getopt.getopt(argv, 'hn:', ['help', 'params='])
    except getopt.GetoptError as err:
        print(err)
        return usage()
    if opt_vals[0][0] in ['-h', '--help']:
        return usage()
    elif opt_vals[0][0] == '-n':
        N_SAMPLES = int(opt_vals[0][1])
        NSAMPLE_PER_PROC = N_SAMPLES // SIZE
    if args[0] == 'x':
        from ofdft_ml.quantum.solver import xsolver
        from ofdft_ml.quantum.utils import xpotential_gen

        fname = opt_vals[1][1]
        with open(fname, 'r') as f:
            for line in f:
                ii = line.index('=')
                if line[:ii]=='n_points': N_POINTS = int(line[(ii+1):])
                if line[:ii]=='n_Gauss': N_GAUSS = int(line[(ii+1):])
                if line[:ii]=='a': 
                    low, up = line[(ii+1):].split(':')
                    LOW_A, HIGH_A = float(low), float(up)
                if line[:ii]=='b':
                    low, up = line[(ii+1):].split(':')
                    LOW_B, HIGH_B = float(low), float(up)
                if line[:ii]=='c':
                    low, up = line[(ii+1):].split(':')
                    LOW_C, HIGH_C = float(low), float(up)
                if line[:ii]=='ne': NE = int(line[(ii+1):])
        param_gen = xpotential_gen(N_POINTS, N_GAUSS, LOW_A, HIGH_A,\
                                   LOW_B, HIGH_B, LOW_C, HIGH_C, RANDOM_SEED)
        # storage
        POTENTIAL_STORAGE = np.zeros((NSAMPLE_PER_PROC, N_POINTS+1))
        DATA_STORAGE = np.zeros((NSAMPLE_PER_PROC, N_POINTS+1))

        for i in range(NSAMPLE_PER_PROC):
            hamilton_mat, Vx = next(param_gen)
            T, density = xsolver(N_POINTS, hamilton_mat, NE)
            DATA_STORAGE[i] = np.array([T, *density])
            POTENTIAL_STORAGE[i, 1:] = Vx

        DATA = None
        POTENTIAL = None
        if ID == 0:
            DATA = np.zeros((N_SAMPLES, N_POINTS+1))
            POTENTIAL = np.zeros((N_SAMPLES, N_POINTS+1))

        comm.Gather(DATA_STORAGE, DATA, root=0)
        comm.Gather(POTENTIAL_STORAGE, POTENTIAL, root=0)

        if ID == 0:
            with open('density_in_x', 'wb') as f:
                pickle.dump(DATA, f)
            with open('potential_in_x', 'wb') as f1:
                pickle.dump(POTENTIAL, f1)

    elif args[0] == 'k':
        from ofdft_ml.quantum.solver import ksolver
        from ofdft_ml.quantum.utils import periodic_potential

        fname = opt_vals[1][1]
        with open(fname, 'r') as f:
            for line in f:
                ii = line.index('=')
                if line[:ii]=='n_gauss': N_GAUSS = int(line[(ii+1):])
                if line[:ii]=='n_cos': N_COS = int(line[(ii+1):])
                if line[:ii]=='n_basis': N_BASIS = int(line[(ii+1):])
                if line[:ii]=='n_kpoints': N_KPOINTS = int(line[(ii+1):])
                if line[:ii]=='occ': OCC = int(line[(ii+1):])
                if line[:ii] == 'V0': 
                    low, up = line[(ii+1):].split(':')
                    LOW_V0, HIGH_V0 = float(low), float(up)
                if line[:ii] == 'Mu':
                    low, up = line[(ii+1):].split(':') 
                    LOW_Mu, HIGH_Mu = float(low), float(up)
                if line[:ii] == 'scale':
                    low, up = line[(ii+1):].split(':')
                    LOW_L, HIGH_L = float(low), float(up)
                if line[:ii] == 'Mu_cos':
                    low, up = line[(ii+1):].split(':') 
                    LOW_Mu_COS, HIGH_Mu_COS = float(low), float(up)

        parameters = {'n_gauss': N_GAUSS, 'n_cos': N_COS, 'low_V0': LOW_V0, 'high_V0': HIGH_V0, 'low_mu': LOW_Mu,
                      'high_mu': HIGH_Mu, 'low_sigma': LOW_L, 'high_sigma': HIGH_L, 'low_mu_c': LOW_Mu_COS, 'high_mu_c': HIGH_Mu_COS}
        param_gen = periodic_potential(N_BASIS, parameters, RANDOM_SEED)

        # storage
        POTENTIAL_STORAGE = np.zeros((NSAMPLE_PER_PROC, N_BASIS+1), dtype=np.complex64)
        DATA_STORAGE = np.zeros((NSAMPLE_PER_PROC, N_BASIS+1), dtype=np.complex64)

        for i in range(NSAMPLE_PER_PROC):
            hamilton_mat, Vq = next(param_gen())
            T, density, mu = ksolver(N_KPOINTS, N_BASIS, hamilton_mat, OCC)
            DATA_STORAGE[i] = np.array([T, *density], dtype=np.complex64)
            POTENTIAL_STORAGE[i] = np.array([mu, *Vq], dtype=np.complex64)

        DATA = None
        POTENTIAL = None
        if ID == 0:
            DATA = np.zeros((N_SAMPLES, N_BASIS+1), dtype=np.complex64)
            POTENTIAL = np.zeros((N_SAMPLES, N_BASIS+1), dtype=np.complex64)

        comm.Gather(DATA_STORAGE, DATA, root=0)
        comm.Gather(POTENTIAL_STORAGE, POTENTIAL, root=0)

        if ID == 0:
            with open('density_in_k', 'wb') as f:
                pickle.dump(DATA, f)
            with open('potential_in_k', 'wb') as f1:
                pickle.dump(POTENTIAL, f1)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
