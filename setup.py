from setuptools import setup, find_packages

from ofdft_ml import __version__

setup(
    name = 'ofdft-ml',
    version = __version__,
    keywords=['OFDFT', 'ML'],
    description = 'orbital free density functional theory empowered by machine learning',
    license = 'MIT License',
    url = 'https://github.com/HamletWantToCode/ofdft-ml.git',
    author = 'Hongbin Ren',
    author_email = 'hongbinrenscu@outlook.com',
    packages = find_packages(),
    py_modules = ['ofdft_ml.ext_math'],
    scripts = [
               'tools/data_process.py', 
               'tools/database.py', 
               'tools/k2x.py', 
               'tools/model_selection.py'
               ],
    platforms = 'any',
    install_requires = ['scikit-learn==0.19.1',
                        'numpy==1.13.3',
                        'scipy==1.1.0',
                        'matplotlib==2.0.2',
                        'mpi4py==2.0.0'
                       ],
    classifiers = [
                   'Programming Language :: Python :: 3.6',
                  ]
)