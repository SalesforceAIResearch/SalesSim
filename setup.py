#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="usersimeval",
    version="0.1.0",
    description="User simulation evaluation tools",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here based on requirements.txt
    ],
    entry_points={
        'console_scripts': [
            'usersimeval=usersimeval.cli:main',
        ],
    },
    python_requires='>=3.6',
)