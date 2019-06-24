# metrics
import numpy as np 

def mean_square_error(outputs, targets):
    if targets.ndim == 2:
        total_mse = np.mean((outputs-targets)**2, axis=0)
        return {'scalar_error': total_mse[0], 'vector_error': np.mean(total_mse[1:])}
    else:
        return np.mean((outputs-targets)**2)