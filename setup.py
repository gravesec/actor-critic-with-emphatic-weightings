from setuptools import setup, find_packages

setup(name='actor-critic-with-emphatic-weightings',
      version='0.0.1',
      url='http://github.com/gravesec/actor-critic-with-emphatic-weightings',
      packages=find_packages(exclude=['notebooks']),
      description=('Source code for the paper "An Off-policy Policy Gradient Theorem Using Emphatic Weightings"')
      )
