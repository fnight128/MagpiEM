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

from test_utils import (
    TestConfig,
    setup_test_logging,
    get_test_data_path,
    log_test_start,
    log_test_success,
    log_test_failure,
    setup_test_environment,
)

setup_test_environment()
from magpiem.io.io_utils import read_single_tomogram, read_emc_tomogram_raw_data, read_emc_mat
from magpiem.processing.classes.cleaner import Cleaner
from magpiem.processing.cpp_integration import clean_tomo_with_cpp
from magpiem.plotting.plotting_utils import create_lattice_plot_from_raw_data

logger = setup_test_logging()

FLIPPED_DATA_FILE = test_root / "data" / "test_data_flipped.mat"
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
        
        tomo_raw_data = read_emc_tomogram_raw_data(full_geom[TEST_TOMO_NAME], TEST_TOMO_NAME)
        if tomo_raw_data is None:
            raise ValueError(f"Failed to extract data for tomogram: {TEST_TOMO_NAME}")
        
        logger.info(f"Loaded raw data with {len(tomo_raw_data)} particles")

        test_cleaner = Cleaner.from_user_params(*TEST_CLEANER_VALUES, allow_flips=True)
        
        logger.info(f"Cleaner allow_flips set to: {test_cleaner.allow_flips}")
        logger.info("Running C++ cleaning with flips enabled...")
        
        lattice_data = clean_tomo_with_cpp(tomo_raw_data, test_cleaner)
        
        logger.info(f"Cleaning completed. Found {len(lattice_data)} lattices")
        
        for lattice_id, particles in lattice_data.items():
            if lattice_id == 0:
                logger.debug(f"Lattice {lattice_id}: {len(particles)} unassigned particles")
            else:
                logger.debug(f"Lattice {lattice_id}: {len(particles)} particles")

        # Generate cone plot directly from raw data and lattice results
        logger.info("Generating cone plot for inspection...")
        fig = create_lattice_plot_from_raw_data(
            tomogram_raw_data=tomo_raw_data,
            lattice_data=lattice_data,
            cone_size=10.0,
            show_removed_particles=False,
        )

        output_html = test_root / "logs" / "cleaning_with_flips_result.html"
        logger.info(f"Saving interactive plot to {output_html}")
        fig.write_html(str(output_html))

        log_test_success(test_name, logger)
        logger.info(f"Cleaning with flips completed successfully")
        logger.info(f"Interactive plot saved as: {output_html}")
        
        # Basic assertions
        assert fig is not None
        assert len(lattice_data) > 1
        assert output_html.exists()

    except Exception as e:
        log_test_failure(test_name, e, logger)
        raise


if __name__ == "__main__":
    test_cleaning_with_flips()
    print("Test completed successfully!")
