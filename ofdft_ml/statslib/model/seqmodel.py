import numpy as np 

class SeqModel(object):
    def __init__(self, gamma=1, noise=0.1, bounds=None, scalar_model=None, vector_model=None):
        self._gamma = gamma
        self._noise = noise
        self.bounds = bounds
        self.scalar_model = scalar_model
        self.vector_model = vector_model

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    def fit(self, train_x, train_y):
        model1 = self.scalar_model(self.gamma, self.noise, self.bounds, self.mode)
        model1.fit(train_x, train_y[:, 0])
        self.gamma = model1.gamma
        self.noise = model1.noise

        n_task = train_y[:, 1:].shape[1]
        model2 = self.vector_model(self.gamma, self.noise, n_task)
        model2.fit(train_x, train_y[:, 1:])

        self.scalar_model = model1
        self.vector_model = model2

    def predict(self, x):
        scalar_target = self.scalar_model.predict(x)
        vector_targets = self.vector_model.predict(x)
        return np.c_[scalar_target[:, np.newaxis], vector_targets]

    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, v):
        if (v>self.bounds[0][0]) and (v<self.bounds[0][1]):
            self._gamma = v
        else:
            raise ValueError
        
    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, v):
        if (v>self.bounds[1][0]) and (v<self.bounds[1][1]):
            self._noise = v
        else:
            raise ValueError