#!/usr/bin/env python3
"""
Simple test to ensure Tomogram class is able to instantiate, run cleaning, and produce a plot
"""

import pathlib
import sys
import logging

# Ensure we use the local magpiem module
current_dir = pathlib.Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from magpiem.read_write import read_single_tomogram
from magpiem.cleaner import Cleaner

# Set up logging
logger = logging.getLogger(__name__)

# Test configuration
TEST_DATA_FILE = current_dir / "WT_CA_2nd.mat"
TEST_TOMO_NAME = "wt2nd_4004_6"
TEST_CLEANER_VALUES = [2.0, 3, 10, 60.0, 40.0, 10.0, 20.0, 90.0, 20.0]


def create_test_cleaner() -> Cleaner:
    """Create a test cleaner with the standard test parameters"""
    return Cleaner.from_user_params(*TEST_CLEANER_VALUES)


def test_tomogram_plotting():
    """Test the refactored tomogram class with plotting_helpers"""
    logger.info("Testing refactored tomogram class with plotting_helpers...")

    # Load test data
    logger.info(f"Loading test data from {TEST_DATA_FILE}")
    test_tomo = read_single_tomogram(str(TEST_DATA_FILE), TEST_TOMO_NAME)
    logger.info(
        f"Loaded tomogram '{test_tomo.name}' with {len(test_tomo.all_particles)} particles"
    )

    test_cleaner = create_test_cleaner()
    test_tomo.set_clean_params(test_cleaner)
    logger.info("Set cleaning parameters")

    logger.info("Running automatic cleaning...")
    test_tomo.autoclean()
    logger.info(
        f"Cleaning completed. Found {len(test_tomo.lattices) - 1} lattices (excluding lattice 0)"
    )

    for lattice_id, particles in test_tomo.lattices.items():
        if lattice_id == 0:
            logger.info(f"Lattice {lattice_id}: {len(particles)} unassigned particles")
        else:
            logger.info(f"Lattice {lattice_id}: {len(particles)} particles")

    logger.info("Generating cone plot...")
    fig = test_tomo.plot_all_lattices(
        cone_size=3.0,
        cone_fix=True,
        showing_removed_particles=False,
    )

    html_file = current_dir / "tomogram_cone_plot.html"
    logger.info(f"Saving interactive plot to {html_file}")
    fig.write_html(str(html_file))

    logger.info("✓ Test completed successfully!")
    logger.info(f"✓ Cone plot saved as: {html_file}")

    return test_tomo, fig


def main():
    """Main test function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    try:
        # Run the main test
        test_tomo, fig = test_tomogram_plotting()

        logger.info("\n✓ All plotting tests completed successfully!")
        logger.info("Tomogram.py successfully produced a cone plot")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
