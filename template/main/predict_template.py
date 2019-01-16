# data predicting

import pickle
import numpy as np 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

from quantum.optimizer import Minimizer
from plot_tool import plot_prediction

with open('../results/demo_best_estimator', 'rb') as f:
    estimator = pickle.load(f)
with open('../results/demo_test_data', 'rb') as f1:
    test_data = pickle.load(f1)
Ek_test, densx_test, dEkx_test = test_data[:, 0], test_data[:, 1:503], test_data[:, 503:]

## estimate the kinetic energy
Ek_predict = estimator.predict(densx_test)

## estimate the kinetic energy derivative
dEkxt_predict = estimator.predict_gradient(densx_test)
dEkx_predict = estimator.named_steps['reduce_dim'].inverse_transform_gradient(dEkxt_predict)

## estimate the ground state electron density
with open('../results/demo_train_data', 'rb') as f2:
    train_data = pickle.load(f2)
Ek_train, densx_train, dEkx_train = train_data[:, 0], train_data[:, 1:503], train_data[:, 503:] 
densx_init = densx_train[0]
densx_true, Vx_true = densx_test[4], -dEkx_test[4]
mu, N = 1.0, 1.0
optimizer = Minimizer(estimator)
densx_predict = optimizer.run(densx_init[np.newaxis, :], Vx_true[np.newaxis, :], mu, N)

## plot results
plot_prediction(Ek_test, Ek_predict, densx_true, densx_predict, densx_init, out_dir='../results')


