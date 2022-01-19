from setuptools import setup, find_packages

setup(
    name='dlrom',
    version='0.1.0',
    packages=find_packages(include=['dlrom', 'dlrom.*']),
    install_requires=[
        'ipython>=7.15.0',
        'numpy>=1.18.1',
        'matplotlib>=3.2.1',
	'fenics==2019.1.0',
	'mshr<=2019.1.0',
	'scipy>=1.4.1',
	'pytorch>=1.5.0'
    ],
    dependency_links=[
        'https://anaconda.org/conda-forge/mshr/2019.1.0/download/linux-64/mshr-2019.1.0-py36hf9f41d3_3.tar.bz2'
    ]
)