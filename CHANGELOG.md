# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Gauss process regression module

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
