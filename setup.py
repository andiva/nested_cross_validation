from setuptools import setup, find_packages
import nested_cross_validation as ncv

setup(
    name='nested_cross_validation',
    python_requires='>3.6',
    version=ncv.__version__,
    packages=find_packages(),
    author='Andrei Ivanov',
    author_email='05x.andrey@gmail.com',
    url='https://github.com/andiva/nested_cross_validation',
    test_suite='tests',
)