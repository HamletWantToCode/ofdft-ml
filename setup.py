from setuptools import setup, find_packages

setup(
    name = 'ofdft-ml',
    version = '0.01',
    keywords=['OFDFT', 'ML'],
    description = 'orbital free density functional theory empowered by machine learning',
    license = 'MIT License',
    url = 'https://github.com/HamletWantToCode/ofdft-ml.git',
    author = 'Hongbin Ren',
    author_email = 'hongbinrenscu@outlook.com',
    package_dir = {'': 'src'},
    packages = find_packages(),
    py_modules = ['database', 'k2x', 'data_process', 'model_selection', 'ofdft-ml.ext_math'],
    platforms = 'any',
    install_requires = ['scikit-learn=0.19.1',
                        'numpy=1.13.3',
                        'scipy=1.1.0',
                        'matplotlib=2.0.2',
                        'mpi4py=2.0.0'
                       ],
    classifiers = [
                   'Programming Language :: Python :: 3.6.5',
                  ]
)