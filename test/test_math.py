import numpy as np 
from ext_math import euclidean_distance

# check distance
X = np.array([[1, 2, 3],
              [4, 5, 6]])

true_eu_dist = np.array([[0, 3*np.sqrt(3)],
                         [3*np.sqrt(3), 0]])

# true_manh_dist = np.array([[0, 9],
#                            [9, 0]])

compute_eu_dist = euclidean_distance(X, X)
# compute_manh_dist = manhattan_distance(X, X)

eu_err = abs(compute_eu_dist - true_eu_dist)
# manh_err = abs(compute_manh_dist - true_manh_dist)

print(eu_err)
# print(manh_err)

