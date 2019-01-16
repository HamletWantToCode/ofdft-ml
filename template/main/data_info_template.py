# template file for data visualizing

import numpy as np 
import matplotlib.pyplot as plt 

from statslib.pca import PrincipalComponentAnalysis
from data_tool import DataTools
from ext_math import euclidean_distance
from plot_tool import *

file_name = '../results/demo_data'
data = DataTools(file_name, PrincipalComponentAnalysis, euclidean_distance)
data.load_data(n_lines=501, start_column=2)

pairwise_distance = data.pairwise_distance()
plot_distance(pairwise_distance, bins=50, out_dir='../results')
principal_components = data.fit_transform(n_components=10)
plot_principal_components(principal_components, hist_params={'bins': 50}, out_dir='../results')

dens_x = data.raw_data_
plot_real_space_density(dens_x, out_dir='../results')
