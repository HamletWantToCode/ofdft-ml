Overview
========

These are the existing features of our code:

* A k-space solver that can solving 1D quantum system with non-interacting electrons given the external periodic potential
* Random periodic potential generator that construct the periodic potential by superposition of cosin function in various magnitude, phase and frequency
* Parallel data generation 
* Euler-Lagrangian solver, constrained optimization method use gradient descent method
* Linear PCA based on SVD algorithm
* Extended kernel ridge regression that enables computing function gradient
* Extended pipeline that enables computing function gradient
* Extended grid search method that measures both the function value and function gradient prediction error
* Matrix inverse & linear equations solver based on SVD algorithm

Features will be released:

* Nonlinear dimension reduction method
* Gauss process regression that support training on function value as well as function gradient
* Hyperparameter searching based on MAP with EM algorithm
* New function gradient error measure
* Parallel grid search process with MPI
* ...
