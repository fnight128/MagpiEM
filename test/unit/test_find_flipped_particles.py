#!/usr/bin/env python3
"""
Test for the find_flipped_particles method in the Tomogram class.

Loads flipped data, cleans, and finds flipped particles, plotting results.
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

from magpiem.io.io_utils import read_single_tomogram
from magpiem.processing.classes.cleaner import Cleaner
from magpiem.plotting.plotting_utils import (
    create_lattice_plot_from_raw_data,
    create_cone_traces,
)
import numpy as np

logger = setup_test_logging()

TEST_DATA_FILE = get_test_data_path("test_data_flipped.mat")
TEST_TOMO_NAME = TestConfig.TEST_TOMO_STANDARD
TEST_CLEANER_VALUES = TestConfig.TEST_CLEANER_VALUES


def test_find_flipped_particles():
    """Test the find_flipped_particles method with visualisation."""
    test_name = "Find Flipped Particles Test"
    log_test_start(test_name, logger)

    try:
        if not TEST_DATA_FILE.exists():
            raise FileNotFoundError(f"Flipped test data not found: {TEST_DATA_FILE}")

        logger.info(f"Loading flipped test data from {TEST_DATA_FILE}")

        test_tomo = read_single_tomogram(str(TEST_DATA_FILE), TEST_TOMO_NAME)
        if test_tomo is None:
            raise ValueError(f"Failed to load tomogram: {TEST_TOMO_NAME}")

        logger.info(f"Loaded tomogram with {len(test_tomo.all_particles)} particles")

        # Create cleaning params manually
        (
            cc_thresh,
            min_neigh,
            min_lattice_size,
            target_dist,
            dist_tol,
            target_ori,
            ori_tol,
            target_curv,
            curv_tol,
        ) = TEST_CLEANER_VALUES

        dist_range = Cleaner.dist_range(target_dist, dist_tol)
        ori_range = Cleaner.ang_range_dotprod(target_ori, ori_tol)
        curv_range = Cleaner.ang_range_dotprod(target_curv, curv_tol)
        flipped_ori_range = tuple(-x for x in reversed(ori_range))

        test_cleaner = Cleaner(
            cc_thresh=cc_thresh,
            min_neighbours=min_neigh,
            min_lattice_size=min_lattice_size,
            dist_range=dist_range,
            ori_range=ori_range,
            curv_range=curv_range,
            allow_flips=True,
            flipped_ori_range=flipped_ori_range,
        )
        test_tomo.cleaning_params = test_cleaner

        logger.info(f"Assigned cleaning parameters: {test_cleaner}")
        logger.info(f"Allow flips: {test_cleaner.allow_flips}")

        logger.info("Cleaning...")
        lattice_data = test_tomo.autoclean()

        logger.info(f"Cleaning complete. Found {len(lattice_data)} lattices")
        for lattice_id, particles in lattice_data.items():
            if lattice_id == 0:
                logger.debug(
                    f"Lattice {lattice_id}: {len(particles)} unassigned particles"
                )
            else:
                logger.debug(f"Lattice {lattice_id}: {len(particles)} particles")

        logger.info("Finding flipped particles...")
        flipped_particles = test_tomo.find_flipped_particles()

        logger.info(f"Found {len(flipped_particles)} flipped particles")
        for particle in flipped_particles:
            logger.debug(f"Flipped particle ID: {particle.particle_id}")

        # Create visualisation with flipped particles highlighted
        logger.info("Visualising...")

        # Create the base lattice plot
        fig = test_tomo.plot_all_lattices(
            cone_size=10.0,
            cone_fix=True,
            showing_removed_particles=False,
        )

        # Add flipped particles in a different colour
        if flipped_particles:
            flipped_positions = []
            flipped_orientations = []

            for particle in flipped_particles:
                flipped_positions.append(particle.position)
                flipped_orientations.append(particle.orientation)

            if flipped_positions:
                flipped_positions = np.array(flipped_positions)
                flipped_orientations = np.array(flipped_orientations)

                flipped_cone_trace = create_cone_traces(
                    positions=flipped_positions,
                    orientations=flipped_orientations,
                    cone_size=2.0,
                    colour="red",
                    opacity=0.9,
                    lattice_id=-1,  # Dummy ID for flipped particles
                )
                flipped_cone_trace.name = "Flipped Particles"

                fig.add_trace(flipped_cone_trace)

        # Save the plot
        output_html = test_root / "logs" / "find_flipped_particles_result.html"
        logger.info(f"Saving interactive plot to {output_html}")
        fig.write_html(str(output_html))

        # Basic assertions
        assert fig is not None, "Figure was not created"
        assert output_html.exists(), "Output HTML file was not created"
        assert isinstance(
            flipped_particles, list
        ), f"Flipped particles returned as an incorrect type, {type(flipped_particles)}"

        total_particles = len(test_tomo.all_particles)
        flipped_count = len(flipped_particles)
        flipped_percentage = (
            (flipped_count / total_particles * 100) if total_particles > 0 else 0
        )

        logger.info(f"Test completed successfully!")
        logger.info(f"Total particles: {total_particles}")
        logger.info(f"Flipped particles: {flipped_count} ({flipped_percentage:.1f}%)")
        logger.info(f"Interactive plot saved as: {output_html}")

        log_test_success(test_name, logger)

    except Exception as e:
        log_test_failure(test_name, e, logger)
        raise


if __name__ == "__main__":
    test_find_flipped_particles()
    print("Test completed successfully!")
