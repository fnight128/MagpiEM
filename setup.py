#!/usr/bin/env python3
"""
Setup script for MagpiEM with automatic C++ compilation
"""

import os
import sys
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy as np


class get_numpy_include(object):
    """Returns Numpy's include directory with lazy import."""

    def __str__(self):
        import numpy

        return numpy.get_include()


class BuildExt(build_ext):
    """Custom build command for C++ extensions."""

    def build_extension(self, ext):
        # Set compiler-specific flags
        if platform.system() == "Windows":
            # Windows-specific flags - simpler configuration
            ext.extra_compile_args = ["/std:c++17"]
            ext.extra_link_args = []
        else:
            # Unix-like systems (Linux, macOS)
            ext.extra_compile_args = ["-std=c++17", "-O2"]
            ext.extra_link_args = []

        super().build_extension(ext)


# Define the C++ extension
processing_extension = Extension(
    "magpiem.processing_cpp",
    sources=[
        "magpiem/cpp/processing_extension.cpp",
    ],
    include_dirs=[
        get_numpy_include(),
        "magpiem/cpp",
    ],
    language="c++",
    define_macros=[
        ("BUILDING_DLL", None),
    ],
)

if __name__ == "__main__":
    setup(
        packages=find_packages(include=["magpiem", "magpiem.*"]),
        ext_modules=[processing_extension],
        cmdclass={"build_ext": BuildExt},
        zip_safe=False,  # Required for C++ extensions
    )
