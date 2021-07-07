# -*- coding: utf-8 -*-

from setuptools import find_packages, setup
setup(
    name='approximationLib',
    packages=find_packages(include=['approximationLib']),
    version='0.1.0',
    description='Approximation Algorithms',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)