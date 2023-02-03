"""
setup.py for VictorNLP Dependency Parsing
"""

import io
from setuptools import find_packages, setup

# Read in the README for the long description on PyPI
def long_description():
    with io.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

# FIXME long_description
setup(name='victornlp_dp',
      version='0.1',
      description='Dependency parsing package',
      long_description="",
      url='https://github.com/jinulee-v/victornlp-dp',
      author='jinulee-v',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3.7'
          ],
      zip_safe=False)