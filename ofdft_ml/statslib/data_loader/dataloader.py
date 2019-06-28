import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, KFold

class Dataset(object):
    def __init__(self, dir_name, test_size):
        # loading dataset
        self.root_name = dir_name
        with open(dir_name+'density_in_k_2x', 'rb') as f1:
            data1 = pickle.load(f1)
        with open(dir_name+'potential_in_k_2x', 'rb') as f2:
            data2 = pickle.load(f2)
        self.features = data1[:, 1:]
        self.feature_dims = self.features.shape[1]
        self.targets = np.c_[data1[:, 0][:, np.newaxis], -1*data2[:, 1:]]  # add on "-1" because dEk ~ -1*V(x)
        self.targets_dims = self.targets.shape[1]
        # additional settings
        self.test_size = test_size

    def train_test(self):
        split_data = train_test_split(self.features, self.targets,
                                      test_size=self.test_size,
                                      shuffle=True, random_state=0)
        self.all_train_features = split_data[0]
        self.test_features = split_data[1]
        self.all_train_targets = split_data[2]
        self.test_targets = split_data[3]

        print('saving training and testing data ......')
        createFolder(self.root_name + 'train')
        np.save(self.root_name+'train/features', self.all_train_features)
        np.save(self.root_name+'train/targets', self.all_train_targets)
        createFolder(self.root_name + 'test')
        np.save(self.root_name+'test/features', self.test_features)
        np.save(self.root_name+'test/targets', self.test_targets)
        print('finish saving !')

    def get_sub_train_set(self, train_size):
        self.all_train_features = self.all_train_features[:train_size]
        self.all_train_targets = self.all_train_targets[:train_size]

    @property
    def len_train(self):
        return len(self.all_train_features)

    @property
    def len_test(self):
        return len(self.test_features)

    @property
    def len_dataset(self):
        return len(self.features)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
