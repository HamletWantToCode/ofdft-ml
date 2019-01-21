# database parallel

import numpy as np
from mpi4py import MPI
import pickle
from absl import app, flags

from quantum.utils import simple_potential_gen
from quantum.solver import solver

FLAGS = flags.FLAGS
flags.DEFINE_integer("N_SAMPLES", 0, "number of data samples in database")
flags.DEFINE_integer("N_COS", 0, "number of cosin functions to sum up")
flags.DEFINE_integer("N_KPOINTS", 0, "number of k points in 1st BZ")
flags.DEFINE_integer("N_BASIS", 0, "number of plane wave basis function used")
flags.DEFINE_float("LOW_V0", 0, "lower bound for potential strength", lower_bound=0)
flags.DEFINE_float("HIGH_V0", 0, "higher bound for potential strength", lower_bound=0)
flags.DEFINE_float("LOW_PHI0", 0, "lower bound for position of the valley of cosin function")
flags.DEFINE_float("HIGH_PHI0", 0, "higher bound for position of the valley of cosin function")

def main(argv):
    del argv

    comm = MPI.COMM_WORLD
    SIZE = comm.Get_size()
    NSAMPLE_PER_PROC = FLAGS.N_SAMPLES // SIZE
    ID = comm.Get_rank()

    POTENTIAL_STORAGE = np.zeros((NSAMPLE_PER_PROC, FLAGS.N_BASIS+1), dtype=np.complex64)
    DATA_STORAGE = np.zeros((NSAMPLE_PER_PROC, FLAGS.N_BASIS+1), dtype=np.complex64)
    RANDOM_STATE = ID
    param_gen = simple_potential_gen(FLAGS.N_BASIS, FLAGS.N_COS, FLAGS.LOW_V0, FLAGS.HIGH_V0,\
                                     FLAGS.LOW_PHI0, FLAGS.HIGH_PHI0, RANDOM_STATE)
    for i in range(NSAMPLE_PER_PROC):
        hamilton_mat, Vq = next(param_gen)
        T, density, mu = solver(FLAGS.N_KPOINTS, FLAGS.N_BASIS, hamilton_mat)
        DATA_STORAGE[i] = np.array([T, *density], dtype=np.complex64)
        POTENTIAL_STORAGE[i] = np.array([mu, *Vq], dtype=np.complex64)

    DATA = None
    POTENTIAL = None
    if ID == 0:
        DATA = np.zeros((FLAGS.N_SAMPLES, FLAGS.N_BASIS+1), dtype=np.complex64)
        POTENTIAL = np.zeros((FLAGS.N_SAMPLES, FLAGS.N_BASIS+1), dtype=np.complex64)

    comm.Gather(DATA_STORAGE, DATA, root=0)
    comm.Gather(POTENTIAL_STORAGE, POTENTIAL, root=0)

    if ID == 0:
        with open('../test/quantum', 'wb') as f:
            pickle.dump(DATA, f)
        with open('../test/potential', 'wb') as f1:
            pickle.dump(POTENTIAL, f1)

if __name__ == '__main__':
    app.run(main)
