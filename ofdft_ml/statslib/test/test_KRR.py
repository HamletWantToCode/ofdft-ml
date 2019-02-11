import pickle
import numpy as np 
import matplotlib.pyplot as plt  

from statslib.kernel_ridge import KernelRidge

fname = '/Users/hongbinren/Documents/program/statslib/toydataset/boston_data'
with open(fname, 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_X, train_y = data[:200, :-1], data[:200, -1]
test_X, test_y = data[-300:, :-1], data[-300:, -1]

model = KernelRidge()
model.fit(train_X, train_y)
predict_y = model.predict(test_X)

err = np.sqrt(np.mean((predict_y - test_y)**2))
print(err)

plt.plot(test_y, predict_y, 'bo')
plt.plot(test_y, test_y, 'r')
plt.show()