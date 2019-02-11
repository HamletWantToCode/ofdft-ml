# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Gauss process regression module 

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
