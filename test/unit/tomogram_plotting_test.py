#!/usr/bin/env python3
"""
Simple test to ensure Tomogram class is able to instantiate, run cleaning, and produce a plot
"""

import sys
from pathlib import Path

# Add test utilities to path
test_root = Path(__file__).parent.parent
sys.path.insert(0, str(test_root))

from test_utils import (  # noqa: E402
    TestConfig,
    setup_test_logging,
    get_test_data_path,
    log_test_start,
    log_test_success,
    log_test_failure,
    setup_test_environment,
)

setup_test_environment()

from magpiem.io.io_utils import read_single_tomogram  # noqa: E402
from magpiem.processing.classes.cleaner import Cleaner  # noqa: E402

logger = setup_test_logging()

TEST_DATA_FILE = get_test_data_path(TestConfig.TEST_DATA_LARGE)
TEST_TOMO_NAME = TestConfig.TEST_TOMO_STANDARD
TEST_CLEANER_VALUES = TestConfig.TEST_CLEANER_VALUES


def create_test_cleaner() -> Cleaner:
    """Create a test cleaner with the standard test parameters"""
    return Cleaner.from_user_params(*TEST_CLEANER_VALUES)


def test_tomogram_plotting():
    """Test tomogram class plotting functionality."""
    test_name = "Tomogram Plotting Test"
    log_test_start(test_name, logger)

    try:
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
                logger.debug(
                    f"Lattice {lattice_id}: {len(particles)} unassigned particles"
                )
            else:
                logger.debug(f"Lattice {lattice_id}: {len(particles)} particles")

        logger.info("Generating cone plot...")
        fig = test_tomo.plot_all_lattices(
            cone_size=10.0,
            cone_fix=True,
            showing_removed_particles=False,
        )

        test_root = Path(__file__).parent.parent
        html_file = test_root / "logs" / "tomogram_cone_plot.html"
        logger.info(f"Saving interactive plot to {html_file}")
        fig.write_html(str(html_file))

        log_test_success(test_name, logger)
        logger.info(f"✓ Cone plot saved as: {html_file}")

        assert test_tomo is not None
        assert fig is not None
        assert len(test_tomo.lattices) > 1

    except Exception as e:
        log_test_failure(test_name, e, logger)
        raise


if __name__ == "__main__":
    test_tomogram_plotting()
    print("✓ Test completed successfully!")
