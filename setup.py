# -*- coding: utf-8 -*-
"""
@author: Frank
"""

from setuptools import setup, find_packages

with open("README.md") as readme:
    long_description = readme.read()

with open("requirements.txt") as reqs:
    requirements = reqs.read()

setup(
    name="magpiem",
    version="0.2.11",
    description="Automated cleaning of sub-tomogram particle picking",
    long_description=long_description,
    author="Frank Nightingale",
    author_email="frank.nightingale@linacre.ox.ac.uk",
    url="https://github.com/fnight128/MagpiEM",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={"console_scripts": ["magpiem = magpiem.dash_ui:main"]},
    python_requires=">3.8.0",
    extras_require={"test": ["pytest"]},
    tests_require=["magpiem[test]"],
)
