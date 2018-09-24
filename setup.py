#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy >= 1.11', 'pandas >= 0.18.0', 'scipy', 'xarray',
                    'netCDF4', 'dask', 'networkx >= 2.0.0']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='loren_frank_data_processing',
    version='0.5.11.dev0',
    license='GPL-3.0',
    description=('Import data from Loren Frank lab'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/loren_frank_data_processing',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
