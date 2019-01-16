# data management

import numpy as np 
from ext_math import euclidean_distance

class DataTools(object):
    def __init__(self, file_name, decomposition, distance_func):
        self.file_name = file_name
        self.decomposition = decomposition
        self.distance_func = distance_func
    
    def load_data(self, n_lines=100, start_column=0):
        import pickle
        with open(self.file_name, 'rb') as f:
            data = pickle.load(f)
        self.raw_data_ = data[:n_lines, start_column:]
        return self

    def pairwise_distance(self):
        distance_ = euclidean_distance(self.raw_data_, self.raw_data_)
        distance_ = np.ravel(distance_)
        distance_ = distance_[distance_!=0]
        return np.unique(distance_)
    
    def fit_transform(self, **transform_params):
        transformer = self.decomposition(**transform_params)
        return transformer.fit_transform(self.raw_data_)

    def irfft(self, **fft_params):
        n = fft_params['n']
        self.raw_data_ = np.fft.irfft(self.raw_data_, **fft_params)*n
        return self
       
