Introduction
============

Installation
------------

**Install OFDFT-ML from GitHub source**:

First, clone OFDFT-ML using ``git``::

	git clone https://github.com/HamletWantToCode/ofdft-ml.git

Then, ``cd`` to the ofdft-ml folder and run the install command (install in develop mode)::

	cd ofdft-ml
	pip install -e .

**NOTE**: installing mpi4py with pip will throw out error, it's recommanded to use conda to install::

	conda install mpi4py 

Requirement
-----------

Required packages:

* numpy 
* scipy
* matplotlib
* mpi4py
* scikit-learn==0.19.1

Python version 3.6

Getting started
---------------

Generate DFT data
^^^^^^^^^^^^^^^^^

To generate density and kinetic energy of electron in a periodic potential, run::

	database.py -n 100 --params quantum_params k

here, the ``-n`` flag indicate size of the dataset, ``--params`` flag should followed by the parameter file "quantum_params", ``k`` tag means we are solving Schrodinger equation in momentum space. The above command will generates a dataset contains 100 data samples, each data sample is a density-kinetic energy pair, the density in dataset is represented by it's Fourier components in momentum space.

The "quantum_params" file is a plain text file with lines::

	n_basis=10    # indicate the number of plane wave basis being used to expand the wavefunction
	n_kpoints=100 # indicate the number of k points sampled in the 1st BZ
	n_cosin=3     # indicate the number of cosin functions used to construct the periodic potential 
	V0=1:10       # indicate the upper & lower bound of magnitude of the cosin functions (positive)
	Phi0=-0.2:0.2 # indicate the upper & lower bound of phase of the cosin functions
	occ=1         # indicate the electron occupation in a unit cell

Alternatively, you can also generate dataset of finite non-periodic potential by solving Schrodinger equation in real space use finite difference method::

	database.py -n 100 --params quantum_params x

the ``x`` tag is used to indicate that we are solving Schrodinger equation in real space.

The "quantum_params" file in this case is::

	n_points=500    # indicate the number of real space grid 
	n_Gauss=3       # indicate the number of Gauss functions used to construct the periodic potential 
	a=1:10          # indicate the upper & lower bound of magnitude of the Gauss functions (positive)
	b=0.4:0.6       # indicate the upper & lower bound of mean value of the Gauss functions
	c=0.03:0.1      # indicate the upper & lower bound of variance of the Gauss functions
	ne=1            # indicate the number of electron

You can also use MPI to speedup the data generating process::

	mpirun -n 4 database.py -n 100 --params quantum_params k

Data preprocessing
^^^^^^^^^^^^^^^^^^

If you use ``k`` tag in the generation process, you have to transform the Fourier components in momentum space back to real space by::

	k2x.py -n 500 -f density_in_k:potential_in_k

here, ``-n`` indicate the number of real space grid used to represent the transformed density function, input files are behind the ``-f`` option, and separated by ':'.

After transforming data back into the real space, you can visualize the density & potential in real space using::

	plot_rawdata.py --f_dens density_in_x --f_Vx potential_in_x

``--f_dens`` and ``--f_Vx`` flags are used to specify density and potential file respectively.

The density in real space are usually of hundred or thousands of dimensions, we can use PCA to visualize the dataset in low dimension::

	data_process.py -f density_in_x proc

the ``proc`` tag is used to indicate the data preprocessing mode.

Model selection
^^^^^^^^^^^^^^^

Usually, machine learing models contain hyperparameters, you will need to specify these parameters properly before actually start learning. In our code, we select those hyperparameters based on maximum likelihood estimation (MLE).

.. math::

	\theta^*=\arg\max_{\theta}P(\boldsymbol{D}|\theta)

To do grid search, you need first specify the hyperparameter set in the "model_params" file::

	n_components=1 # number of PCA components
	gamma=1e-3     # the initial guess for the gamma parameter for RBF kernel
	beta=1e-8      # the initial guess for beta parameter to control the condition of kernel matrix
	params_bounds=1e-5:1e-1:1e-10:1e-6  # searching region for each parameter

and then run::

	model_selection.py --f_dens density_in_x --f_grad potential_in_x -r 0.4 --params model_params
	
``-r`` indicate the ratio of test set's size over the whole dataset's size.

Machine learning & prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cross validation will help us choose the best hyperparamter, and train a machine learning model with those parameters (stored in "best_estimator" and "best_gd_estimator"), it will also generate training and testing data, which are contained in "train_data" and "test_data". Our prediction will use these data files to predict kinetic energy and ground state electron density of a new sample. To do prediction, run::

	data_process.py -f train_data:test_data:best_estimator:best_gd_estimator --params optim_params pred

``pred`` tag means we are predicting, and since we are using gradient descent method to solve Euler Lagrange equation, you need to specify some optimization parameter for prediction, these parameters are written in the "optim_params" file::

	mu=10      # indicate the chemical potential
	n=1        # indicate the electron number
	step=0.01  # indicate the optimization step length
	tol=1e-5   # indicate the tolerance for optimization