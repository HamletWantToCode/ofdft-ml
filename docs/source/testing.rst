Testing
=======

This package contains complete test suits that will help you test the installation and the code, to start testing, you need to install pytest module first, run::

	pip install -U pytest

After the installation finished, ``cd`` into the root directory of this package and run::

	pytest

this command will automatically run all the test functions contained in the package and report bugs.

Known issue
-----------

* To solve linear equations involved in our algorithm, we use a method based on singular value decomposition, this method will fail to reach the required precision if the matrix formed by the coefficients of the linear equaitons is ill-conditioned.
* The accuracy of kinetic energy is related to the number of k points in 1st BZ, to reach high precision, you need more k points.