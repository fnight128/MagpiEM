# -*- coding: utf-8 -*-
"""
C++ integration functions for the MagpiEM application.
"""

import ctypes
import logging
from pathlib import Path

import numpy as np

from .classes import Cleaner

logger = logging.getLogger(__name__)


class CleanParams(ctypes.Structure):
    """C++ structure for cleaning parameters."""

    _fields_ = [
        ("min_distance", ctypes.c_float),
        ("max_distance", ctypes.c_float),
        ("min_orientation", ctypes.c_float),
        ("max_orientation", ctypes.c_float),
        ("min_curvature", ctypes.c_float),
        ("max_curvature", ctypes.c_float),
        ("min_lattice_size", ctypes.c_uint),
        ("min_neighbours", ctypes.c_uint),
    ]


def setup_cpp_library() -> ctypes.CDLL:
    """Load and configure the C++ library."""
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    libname = project_root / "processing" / "processing.dll"

    logger.debug("Loading processing library from: %s", libname)
    logger.debug("Current file: %s", current_file)
    logger.debug("Project root: %s", project_root)

    if not libname.exists():
        alt_paths = [
            project_root / "processing.dll",
            Path.cwd() / "processing" / "processing.dll",
            Path.cwd() / "processing.dll",
        ]

        for alt_path in alt_paths:
            if alt_path.exists():
                libname = alt_path
                logger.info("Found DLL at alternative path: %s", libname)
                break
        else:
            raise FileNotFoundError(
                f"Processing DLL not found. Tried:\n- {libname}\n"
                + "\n".join(f"- {p}" for p in alt_paths)
            )

    c_lib = ctypes.CDLL(str(libname))

    c_lib.clean_particles.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(CleanParams),
        ctypes.POINTER(ctypes.c_int),
    ]
    c_lib.clean_particles.restype = None

    # Set up logging function if available
    if hasattr(c_lib, "set_log_level"):
        c_lib.set_log_level.argtypes = [ctypes.c_int]
        c_lib.set_log_level.restype = None

        # Map Python logging levels to C++ levels
        log_level_map = {
            logging.DEBUG: 0,  # LOG_DEBUG
            logging.INFO: 1,  # LOG_INFO
            logging.WARNING: 2,  # LOG_WARNING
            logging.ERROR: 3,  # LOG_ERROR
        }

        # Get current Python log level and set C++ level accordingly
        current_level = logging.getLogger().getEffectiveLevel()
        cpp_level = log_level_map.get(current_level, 2)  # Default to WARNING
        c_lib.set_log_level(cpp_level)
        logger.debug(
            "Set C++ log level to %d (Python level: %s)", cpp_level, current_level
        )

    return c_lib


def convert_raw_data_to_cpp_format(tomogram_raw_data: list) -> tuple[np.ndarray, int]:
    """
    Convert raw tomogram data to the format expected by the C++ library.

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]

    Returns
    -------
    tuple[np.ndarray, int]
        Flattened data array and number of particles
    """
    if not tomogram_raw_data:
        return np.array([]), 0

    flat_data = []
    for particle in tomogram_raw_data:
        pos, orient = particle
        pos_list = list(pos)
        orient_list = list(orient)
        flat_data.extend(pos_list + orient_list)

    return np.array(flat_data, dtype=np.float32), len(tomogram_raw_data)


def clean_tomo_with_cpp(tomogram_raw_data: list, clean_params: Cleaner) -> dict:
    """
    Clean tomogram data using the C++ library.

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]
    clean_params : Cleaner
        Cleaning parameters

    Returns
    -------
    dict
        Dictionary with particle_id -> lattice_id mappings
    """
    flat_data, num_particles = convert_raw_data_to_cpp_format(tomogram_raw_data)

    if num_particles == 0:
        return {}

    cpp_params = CleanParams(
        min_distance=clean_params.dist_range[0],
        max_distance=clean_params.dist_range[1],
        min_orientation=clean_params.ori_range[0],
        max_orientation=clean_params.ori_range[1],
        min_curvature=clean_params.curv_range[0],
        max_curvature=clean_params.curv_range[1],
        min_lattice_size=clean_params.min_lattice_size,
        min_neighbours=clean_params.min_neighbours,
    )

    c_array = (ctypes.c_float * len(flat_data))(*flat_data)
    results_array = (ctypes.c_int * num_particles)()

    c_lib = setup_cpp_library()
    c_lib.clean_particles(
        c_array, num_particles, ctypes.byref(cpp_params), results_array
    )

    lattice_assignments = {}
    for i in range(num_particles):
        lattice_id = int(results_array[i])
        if lattice_id not in lattice_assignments:
            lattice_assignments[lattice_id] = []
        lattice_assignments[lattice_id].append(i)

    return lattice_assignments
