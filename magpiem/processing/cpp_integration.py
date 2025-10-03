# -*- coding: utf-8 -*-
"""
C++ integration functions for the MagpiEM application.
"""

import logging
from pathlib import Path

import numpy as np

from .classes.cleaner import Cleaner

# python fallbacks incase necessary
from .classes.tomogram import Tomogram
from .classes.particle import Particle

logger = logging.getLogger(__name__)

# Keep cached
_cpp_library = None


def setup_cpp_library():
    """Load and configure the C++ library."""
    global _cpp_library

    # Return cached library if already loaded
    if _cpp_library is not None:
        return _cpp_library

    try:
        # Import the compiled extension
        import magpiem.processing_cpp as cpp_module

        _cpp_library = cpp_module
        logger.info("Successfully loaded compiled C++ extension")
        return _cpp_library
    except ImportError as e:
        logger.error(f"Could not import compiled C++ extension: {e}")
        logger.error("Please ensure the package was installed correctly.")
        raise ImportError(
            "C++ extension not available. This usually means the package wasn't "
            "installed correctly or the C++ component failed to compile. "
            "Try reinstalling."
        ) from e


def clear_cpp_library_cache():
    """Clear the cached C++ library instance. Useful for testing or reloading."""
    global _cpp_library
    _cpp_library = None
    logger.debug("Cleared C++ library cache")


def check_cpp_availability() -> bool:
    """
    Check if the C++ library is available without raising an exception.

    Returns
    -------
    bool
        True if C++ library is available, False otherwise
    """
    try:
        setup_cpp_library()
        return True
    except ImportError:
        return False


def convert_raw_data_to_cpp_format(tomogram_raw_data: list) -> tuple[list, int]:
    """
    Convert raw tomogram data to the format expected by the C++ library.

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]

    Returns
    -------
    tuple[list, int]
        Flattened data list and number of particles
    """
    if not tomogram_raw_data:
        return [], 0

    flat_data = []
    for particle in tomogram_raw_data:
        pos, orient = particle
        pos_list = list(pos)
        orient_list = list(orient)
        flat_data.extend(pos_list + orient_list)

    return flat_data, len(tomogram_raw_data)


def invert_lattice_assignments(results: dict) -> dict:
    """
    Dict inversion to convert from dict(particle_index -> lattice_id) to dict(lattice_id -> [particle_indices])
    """
    lattice_assignments = {}
    for i, lattice_id in enumerate(results):
        lattice_id = int(lattice_id)
        if lattice_id not in lattice_assignments:
            lattice_assignments[lattice_id] = []
        lattice_assignments[lattice_id].append(i)
    return lattice_assignments


def clean_and_detect_flips_with_cpp(
    tomogram_raw_data: list, clean_params: Cleaner
) -> tuple[dict, list]:
    """
    Clean tomogram data and detect flipped particles using the C++ library, with fallback to python

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]
    clean_params : Cleaner
        Cleaning parameters

    Returns
    -------
    tuple[dict, list]
        Dictionary with lattice_id -> [particle_indices] mappings and list of flipped particle indices
    """
    if not tomogram_raw_data:
        return {}, []

    try:
        # Try to use C++ implementation first
        return _clean_and_detect_flips_with_cpp(tomogram_raw_data, clean_params)
    except ImportError as e:
        logger.warning(f"C++ library not available: {e}")
        logger.info("Falling back to Python implementation using Tomogram class")
        return _clean_and_detect_flips_with_python_fallback(
            tomogram_raw_data, clean_params
        )


def clean_tomo_with_cpp(tomogram_raw_data: list, clean_params: Cleaner) -> dict:
    """
    Clean tomogram data using the C++ library, with fallback to python

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]
    clean_params : Cleaner
        Cleaning parameters

    Returns
    -------
    dict
        Dictionary with lattice_id -> [particle_indices] mappings
    """
    if not tomogram_raw_data:
        return {}

    try:
        # Try to use C++ implementation first
        return _clean_with_cpp(tomogram_raw_data, clean_params)
    except ImportError as e:
        logger.warning(f"C++ library not available: {e}")
        logger.info("Falling back to Python implementation using Tomogram class")
        return _clean_with_python_fallback(tomogram_raw_data, clean_params)


def _clean_and_detect_flips_with_cpp(
    tomogram_raw_data: list, clean_params: Cleaner
) -> tuple[dict, list]:
    """Clean and detect flips using C++ library implementation."""
    logger.debug(f"Python wrapper called with {len(tomogram_raw_data)} particles")
    flat_data, num_particles = convert_raw_data_to_cpp_format(tomogram_raw_data)
    logger.debug(f"Converted to {num_particles} particles")

    if num_particles == 0:
        return {}, []

    # Convert parameters to Python list format expected by the extension
    params_list = [
        clean_params.dist_range[0],  # min_distance
        clean_params.dist_range[1],  # max_distance
        clean_params.ori_range[0],  # min_orientation
        clean_params.ori_range[1],  # max_orientation
        clean_params.curv_range[0],  # min_curvature
        clean_params.curv_range[1],  # max_curvature
        clean_params.min_lattice_size,
        clean_params.min_neighbours,
        clean_params.allow_flips,
    ]
    logger.debug(f"Parameters: {params_list}")

    c_lib = setup_cpp_library()
    logger.debug("Calling C++ clean_and_detect_flips function")

    lattice_results, flipped_results = c_lib.clean_and_detect_flips(
        flat_data, num_particles, params_list
    )
    logger.debug(
        f"C++ returned {len(flipped_results)} results, {sum(flipped_results)} flipped"
    )
    logger.debug(f"Lattice results: {lattice_results}")
    logger.debug(f"Flipped results: {flipped_results}")

    # Convert lattice results to dictionary format
    lattice_assignments = invert_lattice_assignments(lattice_results)

    # Convert flipped results to list of particle indices
    flipped_particles = [
        i for i, is_flipped in enumerate(flipped_results) if is_flipped
    ]

    return lattice_assignments, flipped_particles


def debug_flip_detection_with_cpp(
    tomogram_raw_data: list, clean_params: Cleaner
) -> tuple[dict, list]:
    """
    Debug flip detection using C++ library with manual lattice assignment.

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]
    clean_params : Cleaner
        Cleaning parameters

    Returns
    -------
    tuple[dict, list]
        Dictionary with lattice_id -> [particle_indices] mappings and list of flipped particle indices
    """
    if not tomogram_raw_data or not clean_params:
        raise ValueError("Empty data provided")

    try:
        return _debug_flip_detection_with_cpp(tomogram_raw_data, clean_params)
    except ImportError as e:
        logger.error(f"C++ library not available: {e}")
        logger.info("Falling back to Python implementation using Tomogram class")
        return _clean_and_detect_flips_with_python_fallback(
            tomogram_raw_data, clean_params
        )


def _debug_flip_detection_with_cpp(
    tomogram_raw_data: list, clean_params: Cleaner
) -> tuple[dict, list]:
    """Debug flip detection using C++ library implementation."""
    flat_data, num_particles = convert_raw_data_to_cpp_format(tomogram_raw_data)

    if num_particles == 0:
        return {}, []

    # Convert parameters to Python list format expected by the extension
    params_list = [
        clean_params.dist_range[0],  # min_distance
        clean_params.dist_range[1],  # max_distance
        clean_params.ori_range[0],  # min_orientation
        clean_params.ori_range[1],  # max_orientation
        clean_params.curv_range[0],  # min_curvature
        clean_params.curv_range[1],  # max_curvature
        clean_params.min_lattice_size,
        clean_params.min_neighbours,
        clean_params.allow_flips,
    ]

    c_lib = setup_cpp_library()

    lattice_results, flipped_results = c_lib.debug_flip_detection(
        flat_data, num_particles, params_list
    )

    # Convert lattice results to dictionary format
    lattice_assignments = invert_lattice_assignments(lattice_results)

    # Convert flipped results to list of particle indices
    flipped_particles = [
        i for i, is_flipped in enumerate(flipped_results) if is_flipped
    ]

    return lattice_assignments, flipped_particles


def _clean_and_detect_flips_with_python_fallback(
    tomogram_raw_data: list, clean_params: Cleaner
) -> tuple[dict, list]:
    """Fallback to Python implementation for cleaning and flip detection."""
    # Create a temporary tomogram for processing
    temp_tomo = Tomogram("temp")
    temp_tomo.all_particles = Particle.from_array(tomogram_raw_data, temp_tomo)
    temp_tomo.cleaning_params = clean_params

    lattice_data = temp_tomo.autoclean()
    flipped_particles = temp_tomo.find_flipped_particles()

    # Convert flipped particles to indices
    flipped_indices = [p.particle_id for p in flipped_particles]

    return lattice_data, flipped_indices


def _clean_with_cpp(tomogram_raw_data: list, clean_params: Cleaner) -> dict:
    """Clean using C++ library implementation."""
    flat_data, num_particles = convert_raw_data_to_cpp_format(tomogram_raw_data)

    if num_particles == 0:
        return {}

    # Convert parameters to Python list format expected by the extension
    params_list = [
        clean_params.dist_range[0],  # min_distance
        clean_params.dist_range[1],  # max_distance
        clean_params.ori_range[0],  # min_orientation
        clean_params.ori_range[1],  # max_orientation
        clean_params.curv_range[0],  # min_curvature
        clean_params.curv_range[1],  # max_curvature
        clean_params.min_lattice_size,
        clean_params.min_neighbours,
        clean_params.allow_flips,
    ]

    c_lib = setup_cpp_library()

    # call with Python lists
    results = c_lib.clean_particles(flat_data, num_particles, params_list)

    return invert_lattice_assignments(results)


def _clean_with_python_fallback(tomogram_raw_data: list, clean_params: Cleaner) -> dict:
    """Clean using Python Tomogram class as fallback."""

    temp_tomogram = Tomogram("temp_fallback")
    temp_tomogram.set_clean_params(clean_params)

    # Convert raw data to format expected by Particle.from_array
    # Format: [cc_value, [x, y, z], [u, v, w]]
    particle_data_list = []
    for pos, orient in tomogram_raw_data:
        particle_data_list.append(
            [10.0, pos, orient]
        )  # High CC score to pass filtering

    particles = Particle.from_array(particle_data_list, temp_tomogram)
    temp_tomogram.assign_particles(particles)

    result_dict = temp_tomogram.autoclean()

    if result_dict is None:
        logger.warning(
            "No lattices found after cleaning - all particles may have been filtered out"
        )
        return {}

    lattice_assignments = {}
    for lattice_id, particle_ids in result_dict.items():
        if lattice_id != "selected":
            lattice_assignments[int(lattice_id)] = particle_ids

    logger.debug(f"Final lattice assignments: {lattice_assignments}")

    return lattice_assignments
