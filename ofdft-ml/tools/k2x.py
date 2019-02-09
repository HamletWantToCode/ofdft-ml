import pickle
import sys
import getopt
import numpy as np 

def main(argv, N_POINTS=100):
    # parsing parameters
    def usage():
        print("""
              usage:
              -h: help
              -f: input files, seperated by ':'
              """)
        return 100
    try:
        opt_vals, _ = getopt.getopt(argv, 'hf:', ['help', 'file='])
    except getopt.GetoptError as err:
        print(err)
        return usage()
    for (opt, vals) in opt_vals:
        if opt in ['-h', '--help']: return usage()
        elif opt in ['-f', '--file']: file_list = vals.split(':')
    # read and transform files
    for fname in file_list:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        scaler, vec_q = data[:, 0].real, data[:, 1:]
        vec_x = np.fft.irfft(vec_q, axis=1, n=N_POINTS)*N_POINTS
        vec_x = vec_x.real
        data_t = np.c_[scaler.reshape((-1, 1)), vec_x]
        with open('_'.join([fname, '2x']), 'wb') as f2:
            pickle.dump(file=f2, obj=data_t)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))