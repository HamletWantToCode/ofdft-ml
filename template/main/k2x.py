import pickle
import numpy as np 

data_file = '../results/quantum'
Vq_file = '../results/potential'

with open(data_file, 'rb') as f:
    data_q = pickle.load(f)
data_x = np.fft.irfft(data_q[:, 1:], axis=1, n=100)*100
data_x = data_x.real
Ek = data_q[:, 0].real
new_data = np.c_[Ek.reshape((-1, 1)), data_x]

with open(Vq_file, 'rb') as f1:
    Vq = pickle.load(f1)
Vx = np.fft.irfft(Vq[:, 1:], axis=1, n=100)*100
mu = Vq[:, 0].real
Vx = Vx.real
new_potential = np.c_[mu.reshape((-1, 1)), Vx]

with open('../results/demo_data', 'wb') as f2:
    pickle.dump(file=f2, obj=new_data)

with open('../results/demo_Vx', 'wb') as f3:
    pickle.dump(file=f3, obj=new_potential)
