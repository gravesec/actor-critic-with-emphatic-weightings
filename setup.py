from setuptools import setup, find_packages

setup(name='off-policy-actor-critic',
      version='0.1',
      url='http://github.com/gravesec/off-policy-actor-critic',
      packages=find_packages(exclude=['test','test.*']),
      )
