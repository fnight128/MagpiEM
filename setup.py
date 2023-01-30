# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:05:54 2023

@author: Frank
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.readlines()

long_description = "Sample Package made for a demo \
      of its making for the GeeksforGeeks Article."

setup(
    name="MagpiEM",
    version="0.1",
    description="Automated cleaning of sub-tomogram particle picking",
    author="Frank Nightingale",
    author_email="frank.nightingale@linacre.ox.ac.uk",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={"console_scripts": ["magpiem = MagpiEM.dash_ui:main"]},
    python_requires=">3.8.0",
)
