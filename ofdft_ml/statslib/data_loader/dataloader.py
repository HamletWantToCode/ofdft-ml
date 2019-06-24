import pickle
from sklearn.model_selection import train_test_split

class Dataset(object):
    def __init__(self, dir_name, test_size, valid_size=None, n_cv=None):
        # loading dataset 
        with open(dir_name+'density_in_k_2x', 'rb') as f1:
            data1 = pickle.load(f1)
        with open(dir_name+'potential_in_k_2x', 'rb') as f2:
            data2 = pickle.load(f2)
        self.features = data1[:, 1:]
        self.feature_dims = self.features.shape[1]
        self.targets = np.c_[data1[:, 0][:, np.newaxis], data2[:, 1:]]
        self.targets_dims = self.targets.shape[1]
        # additional settings
        self.test_size = test_size
        self.valid_size = valid_size
        self.n_cv = n_cv

    def train_test(self):
        split_data = train_test_split(self.features, self.targets,
                                      test_size=self.test_size,
                                      shuffle=True, random_state=0)
        self.all_train_features = split_data[0]
        self.test_features = split_data[1]
        self.all_train_targets = split_data[2]
        self.test_targets = split_data[3]
    
    def train_validate(self):
        if hasattr(self, 'all_train_features') and hasattr(self, 'all_train_targets'):
            split_data = train_test_split(self.all_train_features,
                                          self.all_train_targets,
                                          test_size=self.valid_size,
                                          shuffle=True, random_state=1)
            self.train_features = split_data[0]
            self.valid_features = split_data[1]
            self.train_targets = split_data[2]
            self.valid_targets = split_data[3]
        else:
            print('Please do train test splitting first !')

    def cross_validate(self):
        pass
