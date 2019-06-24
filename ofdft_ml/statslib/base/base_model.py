from abc import abstractmethod

class BaseGP(object):
    def __init__(self, noise, kernel, mean_f):
        self.noise = noise
        self.kernel = kernel
        self.mean_f = mean_f if callable(mean_f) else None

    @abstractmethod
    def constraint_optimizer(self, variables, obj_fun, obj_gradient):
        raise NotImplementedError

    def fit(self, train_x, train_y):
        pass

    def predict(self, x):
        pass




