# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- 'statslib.GaussProcess': add multiple startups, able to jump out of local minimum

## 0.5.2 - 2019-03-19
### Fixed
- 'statslib.GaussProcess': incorrect matrix reshape operation

## 0.5.1 - 2019-03-16
### Added
- 'quantum.utils': use superposition of Gauss function instead of cosin functions, directly sample from Fourier transform of Gauss

### Changed
- 'tools/model_selection': combine grid search and optimization method, much efficient than before

### Removed
- 'statslib.new_scorer'
- 'statslib.new_grid_search'
- 'statslib.kernel_ridge'
- 'tools/data_centre'

### Fixed
- 'statslib.utils': fix the computation error in "rbf_kernel_2nd_gradient"

## 0.5.0 - 2019-03-11
### Added
- 'GaussProcess': implement Bayes method with Gauss process prior as a function fitting tool; hyperparameters are choose by optimize the marginal likelihood
- 'statslib.utils': add kernel function for derivative of gaussian process
- 'test_gp': test suit for Gauss process regressor

### Changed
- 'test_KRR': split the original file into 'test_KRR' and 'test_kernel'
- 'ext_math': SVD or Cholesky decomposition will be done out side the sovler function

## 0.4.0 - 2019-02-25 
### Changed
- 'NewGridSearchCV' based on multiprocessing not run in parallel, change it's implementation to MPI

### Fixed
- Fixed bug in 'plot_rawdata', this bug will throw an error if you only feed Vx data into the program

## 0.3.0 - 2019-02-20
### Added
- Add tool 'plot_rawdata' to plot the density and potential distribution in real space
- Add tool 'data_centre', data will be centred before they are fed into PCA, this should remove the additional pattern introduced by translate V(x)/density along x-axis

### Fixed
- Error when call 'fit_transform' method in pipeline, delete '**fit_params' in arguments

## 0.2.2 - 2019-02-17
### Added
- Implement 'special_potential_gen', a periodic potential with two Gauss peaks 

## 0.2.0 - 2019-02-17
### Added
- New version of grid search CV:
  - clean code
  - support various CV spliting strategies
  - more efficient parallelism (use multiprocessing module)
  - fully compatible with scikit-learn
- New version of pipeline
  - clean code (remove joblib memory inside the code)
  - fully compatible with scikit-learn 
- Added test for grid search CV and pipeline

### Removed
- Remove unused keyword 'kernel_hessan' in "KernelRidge" method

### Changed
- Modify "scorer" module: use sklearn 'make_scorer' implementation, extend function '_PredictScorer' to measure error of function gradient
- "model_selection" script: modify the import of old CV & pipe & scorer

### Deprecated
- Old grid search method will be deprecated in the future
- Old pipeline method will be deprecated in the future

## 0.1.0 - 2019-02-14
### Added
- added "new_grid_search" module

## 0.0.5 - 2019-02-13
### Fixed
- "tools/model_selection.py": in the line 76, the original 'best_score_index' is incorrect (used grid_searchCV 'rank_test_score'), change to 'best_index_' method

## 0.0.4 - 2019-02-12
### Fixed
- "tools/k2x.py": the '-n' tag doesn't work, fixed by adding ':' behind

## 0.0.3 - 2019-02-12
### Fixed
- python scripts in tools/ can't execute in linux system: add "#!/usr/bin/env python"

## 0.0.2 - 2019-02-11
### Added
- New test suit:
  - Tests for "ksolver": energy band calculation & kinetic energy calculation
  - Tests for "xsolver": harmonic oscillator & finite well
  - Tests for "ext_math": linear equation solver & Euclidean distance calculator
  - scikit-learn tests for "statslib"

### Changed
- "EL_solver": change *V* in "energy_gd" function to projected *V_proj*
- "kspace": remove the default *occ* value for "ksolver"

### Issues
- Test for "xsolver": harmonic oscillator energy level difference not accurately (only within 2 digits, expect 7 digits) match the oscillator frequency
- Test for "ext_math": linear solver not accurate (only within 5 digits, expect 7 digits) for solving ill-condition linear system
- Test for "kernel_ridge": the gradient calculation not accurate (only within 4 digits, expect 7 digits) compared to finite difference result

## 0.0.1 - 2019-02-10
### Added
- New "quantum" subpackage:
    - New "solver" module:
        - Solve Schrodinger equation in real grid with fixed boundary condition 
        - Solve Schrodinger equation in k grid with periodic boundary condition 
    - New "util" module contains random potential generator:
        - Aperiodic potential in real grid: superposition of several random Gaussian function
        - Periodic potential in k grid: Superposition of several random cosin function
    - EL equation solver using gradient descent method
- New "statslib" subpackage:
    - New "kernel_ridge" module supports *function gradient* estimation
    - New "pca" module supports tansform&inverse transform of *function gradient*
    - New "pipeline" module supports *function gradient* estimation
    - New "grid_search" module supports measure function estimation error and *function gradient* estimation error simultaneously
- New "ext_math":
  - Various linear equation solver (SVD, Cholesky)
  - Euclidean distance between data samples
- New command line tools ("tools"):
  - "database": generate (density, Ek)&(mu, potential) dataset in real grid or k grid
  - "k2x": fft k grid density/potential to real grid
  - "model_selection": grid search for optimal hyperparameters
  - "data_process": visulize the density data distribution and performance of ML algorithm
