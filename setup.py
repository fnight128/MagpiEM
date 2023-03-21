# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:05:54 2023

@author: Frank
"""

from setuptools import setup, find_packages

requirements = [
    "numpy",
    "pandas",
    "prettytable",
    "scipy",
    "plotly",
    "pathlib",
    "dash==2.8.0",
    "dash_bootstrap_components",
    "dash_bootstrap_templates",
    "dash_extensions",
    "dash_daq",
    "flask",
    "imodmodel",
    "emfile",
    "pyyaml",
]

setup(
    name="MagpiEM",
    version="0.2.7",
    description="Automated cleaning of sub-tomogram particle picking",
    author="Frank Nightingale",
    author_email="frank.nightingale@linacre.ox.ac.uk",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={"console_scripts": ["magpiem = MagpiEM.dash_ui:main"]},
    python_requires=">3.8.0",
)
