#!/usr/bin/env python3
"""
Test for cleaning functionality with particle flipping enabled.

Loads test_data_flipped.mat and runs cleaning with allow_flips enabled,
saving the result as an HTML file for inspection.
"""

import sys
from pathlib import Path

# Add test utilities to path
test_root = Path(__file__).parent.parent
sys.path.insert(0, str(test_root))

from test_utils import (  # noqa: E402
    TestConfig,
    setup_test_logging,
    log_test_start,
    log_test_success,
    log_test_failure,
    setup_test_environment,
    ensure_test_data_generated,
    get_test_data_path,
)

setup_test_environment()
from magpiem.io.io_utils import (  # noqa: E402
    read_single_tomogram,
    read_emc_tomogram_raw_data,
    read_emc_mat,
)
from magpiem.processing.classes.cleaner import Cleaner  # noqa: E402
from magpiem.processing.cpp_integration import (  # noqa: E402
    clean_tomo_with_cpp,
    clean_and_detect_flips_with_cpp,
)
from magpiem.plotting.plotting_utils import (  # noqa: E402
    create_lattice_plot_from_raw_data,
)

logger = setup_test_logging()

# Ensure test data is generated before running tests
ensure_test_data_generated()

FLIPPED_DATA_FILE = get_test_data_path(TestConfig.TEST_DATA_SMALL_FLIPPED)
TEST_TOMO_NAME = TestConfig.TEST_TOMO_STANDARD
TEST_CLEANER_VALUES = TestConfig.TEST_CLEANER_VALUES


def test_cleaning_with_flips():
    """Test cleaning functionality with particle flipping enabled."""
    test_name = "Cleaning with Flips Test"
    log_test_start(test_name, logger)

    try:
        # Check if flipped data exists
        if not FLIPPED_DATA_FILE.exists():
            raise FileNotFoundError(f"Flipped test data not found: {FLIPPED_DATA_FILE}")

        logger.info(f"Loading flipped test data from {FLIPPED_DATA_FILE}")

        # Load the raw tomogram data for C++ processing

        full_geom = read_emc_mat(str(FLIPPED_DATA_FILE))
        if full_geom is None:
            raise ValueError("Failed to load flipped mat file")

        tomo_raw_data = read_emc_tomogram_raw_data(
            full_geom[TEST_TOMO_NAME], TEST_TOMO_NAME
        )
        if tomo_raw_data is None:
            raise ValueError(f"Failed to extract data for tomogram: {TEST_TOMO_NAME}")

        logger.info(f"Loaded raw data with {len(tomo_raw_data)} particles")

        test_cleaner = Cleaner.from_user_params(*TEST_CLEANER_VALUES, allow_flips=True)

        logger.info(f"Cleaner allow_flips set to: {test_cleaner.allow_flips}")

        # Run C++ cleaning with flips enabled
        logger.info("Running C++ cleaning with flips enabled...")
        cpp_lattice_data = clean_tomo_with_cpp(tomo_raw_data, test_cleaner)
        logger.info(f"C++ cleaning completed. Found {len(cpp_lattice_data)} lattices")

        # Run C++ cleaning and flip detection
        logger.info("Running C++ cleaning and flip detection...")
        cpp_flip_lattice_data, cpp_flipped_indices = clean_and_detect_flips_with_cpp(
            tomo_raw_data, test_cleaner
        )
        logger.info(
            f"C++ flip detection completed. Found {len(cpp_flip_lattice_data)} "
            f"lattices and {len(cpp_flipped_indices)} flipped particles"
        )

        # Run Python implementation for comparison
        logger.info("Running Python implementation for comparison...")
        test_tomo = read_single_tomogram(str(FLIPPED_DATA_FILE), TEST_TOMO_NAME)
        if test_tomo is None:
            raise ValueError(f"Failed to load tomogram: {TEST_TOMO_NAME}")

        test_tomo.cleaning_params = test_cleaner
        test_tomo.find_particle_neighbours()

        # Manually assign all particles to lattice 1 for Python comparison
        for particle in test_tomo.all_particles:
            particle.set_lattice(1)
        test_tomo.lattices[1] = test_tomo.all_particles

        # Run Python cleaning
        test_tomo.autoclean()
        python_lattice_data = {
            lattice_id: list(particles)
            for lattice_id, particles in test_tomo.lattices.items()
        }
        logger.info(
            f"Python cleaning completed. Found {len(python_lattice_data)} lattices"
        )

        # Run Python flip detection
        python_flipped_particles = test_tomo.find_flipped_particles()
        python_flipped_indices = [
            particle.particle_id for particle in python_flipped_particles
        ]
        logger.info(
            f"Python flip detection completed. Found "
            f"{len(python_flipped_indices)} flipped particles"
        )

        # Log results for comparison
        for lattice_id, particles in cpp_lattice_data.items():
            if lattice_id == 0:
                logger.debug(
                    f"C++ Lattice {lattice_id}: {len(particles)} unassigned particles"
                )
            else:
                logger.debug(f"C++ Lattice {lattice_id}: {len(particles)} particles")

        for lattice_id, particles in python_lattice_data.items():
            if lattice_id == 0:
                logger.debug(
                    f"Python Lattice {lattice_id}: {len(particles)} unassigned particles"
                )
            else:
                logger.debug(f"Python Lattice {lattice_id}: {len(particles)} particles")

        # Generate comparison plots
        logger.info("Generating comparison plots...")

        # C++ cleaning plot
        cpp_fig = create_lattice_plot_from_raw_data(
            tomogram_raw_data=tomo_raw_data,
            lattice_data=cpp_lattice_data,
            cone_size=10.0,
            show_removed_particles=False,
        )
        cpp_output_html = test_root / "logs" / "cleaning_with_flips_cpp_result.html"
        logger.info(f"Saving C++ interactive plot to {cpp_output_html}")
        cpp_fig.write_html(str(cpp_output_html))

        # Convert Particle objects to indices for plotting
        python_lattice_data_indices = {}
        for lattice_id, particles in python_lattice_data.items():
            python_lattice_data_indices[lattice_id] = [
                particle.particle_id for particle in particles
            ]

        # Python cleaning plot
        python_fig = create_lattice_plot_from_raw_data(
            tomogram_raw_data=tomo_raw_data,
            lattice_data=python_lattice_data_indices,
            cone_size=10.0,
            show_removed_particles=False,
        )
        python_output_html = (
            test_root / "logs" / "cleaning_with_flips_python_result.html"
        )
        logger.info(f"Saving Python interactive plot to {python_output_html}")
        python_fig.write_html(str(python_output_html))

        # Combined comparison plot
        combined_fig = create_lattice_plot_from_raw_data(
            tomogram_raw_data=tomo_raw_data,
            lattice_data=cpp_lattice_data,
            cone_size=10.0,
            show_removed_particles=False,
        )
        combined_output_html = (
            test_root / "logs" / "cleaning_with_flips_combined_result.html"
        )
        logger.info(f"Saving combined interactive plot to {combined_output_html}")
        combined_fig.write_html(str(combined_output_html))

        log_test_success(test_name, logger)
        logger.info("Cleaning with flips completed successfully")
        logger.info(f"C++ interactive plot saved as: {cpp_output_html}")
        logger.info(f"Python interactive plot saved as: {python_output_html}")
        logger.info(f"Combined interactive plot saved as: {combined_output_html}")

        # Log comparison results
        logger.info(
            f"C++ found {len(cpp_lattice_data)} lattices, "
            f"{len(cpp_flipped_indices)} flipped particles"
        )
        logger.info(
            f"Python found {len(python_lattice_data)} lattices, "
            f"{len(python_flipped_indices)} flipped particles"
        )
        logger.info(
            f"Flipped particles match: {set(cpp_flipped_indices) == set(python_flipped_indices)}"
        )

        # All particles should be assigned to lattice 1 as flips are allowed
        assert len(cpp_lattice_data) == 1
        assert len(python_lattice_data) == 1

        # Basic assertions
        assert cpp_fig is not None
        assert python_fig is not None
        assert combined_fig is not None
        assert cpp_output_html.exists()
        assert python_output_html.exists()
        assert combined_output_html.exists()

    except Exception as e:
        log_test_failure(test_name, e, logger)
        raise


if __name__ == "__main__":
    test_cleaning_with_flips()
    logger.info("Test completed successfully!")
