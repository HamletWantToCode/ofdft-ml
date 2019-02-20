import sys
import pickle
import getopt
import numpy as np

def centre(X):
    n_points = X.shape[1]
    mid = n_points // 2
    pos = mid - np.argmax(abs(X), axis=1)
    X_new = np.zeros_like(X)
    n_samples = len(X)
    for i in range(n_samples):
        X_new[i] = np.roll(X[i], shift=pos[i])
    return X_new

def main(argv):
    def usage():
        print(
            """
            usage:
                -h (--help)
                -f: data files need to be centred 
            """
        )
        return 100
    try:
        opt_vals, _ = getopt.getopt(argv, 'hf:', ['help'])
    except getopt.GetoptError as e:
        return usage()
    for (opt, val) in opt_vals:
        if opt in ['-h', '--help']: return usage()
        elif opt == '-f':
            fnames = val.split(':')
            for fname in fnames:
                with open(fname, 'rb') as f:
                    data = pickle.load(f)
                centred_data = centre(data[:, 1:])
                centred_data = np.c_[data[:, 0].reshape((-1, 1)), centred_data]
                with open(fname+'_centred', 'wb') as f:
                    pickle.dump(centred_data, f)
        else:
            usage()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

