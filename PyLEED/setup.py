from __future__ import annotations # Python 3.7+ required
from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='PyLEED',
    version='0.1',
    description='LEED structure determination using Bayesian Optimization',
    long_description=long_description,
    author='Cole Miles',
    author_email='cmm572@cornell.edu',
    packages=['pyleed'],
    install_requires=['numpy', 'matplotlib', 'torch', 'gpytorch', 'botorch', 'IPython'],
    python_requires='>=3.7',
    extras_require={'tests': ['pytest']},
)