#!/usr/bin/env python3
"""
Test for the flip_emc_particles function in io_utils.

Loads test_data.mat, randomly selects 20% of particles to flip,
applies the flip_emc_particles function, and saves the result to test_data_flipped.mat.
"""

import sys
import random
import numpy as np
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

from magpiem.io.io_utils import read_emc_mat, write_emc_mat  # noqa: E402

logger = setup_test_logging()

TEST_DATA_FILE = get_test_data_path(TestConfig.TEST_DATA_STANDARD)

# proportion of particles to flip, chosen randomly
AMOUNT_TO_FLIP = 0.3


def test_flip_emc_particles():
    """Test the write_emc_mat function with particle flipping functionality."""
    test_name = "Flip EMC Particles Test"
    log_test_start(test_name, logger)

    try:
        logger.info(f"Loading test data from {TEST_DATA_FILE}")

        mat_geom = read_emc_mat(str(TEST_DATA_FILE))

        if mat_geom is None:
            raise ValueError("Failed to load mat file")

        logger.info(f"Loaded geometry data with {len(mat_geom)} tomograms")

        particles_to_flip = {}
        total_particles = 0
        flipped_count = 0

        # Choose random particles to flip in each tomogram
        for tomo_id, particles in mat_geom.items():
            if len(particles) == 0:
                continue

            total_particles += len(particles)
            num_to_flip = max(1, int(len(particles) * AMOUNT_TO_FLIP))
            particle_indices = list(range(len(particles)))
            selected_indices = random.sample(particle_indices, num_to_flip)

            particles_to_flip[tomo_id] = selected_indices
            flipped_count += len(selected_indices)

            logger.info(
                f"Tomogram {tomo_id}: {len(particles)} particles, flipping {len(selected_indices)}"
            )

        logger.info(
            f"Total particles: {total_particles}, flipping {flipped_count} ({flipped_count/total_particles*100:.1f}%)"
        )

        # Create keep_ids dict for all particles (since we want to keep all particles, just with some flipped)
        keep_ids = {}
        for tomo_id, particles in mat_geom.items():
            keep_ids[tomo_id] = list(range(len(particles)))

        output_file = test_root / "data" / "test_data_flipped.mat"
        logger.info(f"Saving flipped data to {output_file}")

        write_emc_mat(
            keep_ids,
            str(output_file),
            str(TEST_DATA_FILE),
            flip_particles=particles_to_flip,
        )

        assert output_file.exists(), f"Output file {output_file} was not created"

        # Load created file for some verification
        logger.info("Verifying flip results...")
        flipped_mat_geom = read_emc_mat(str(output_file))

        # Ensure no particles were lost
        for tomo_id, particles in flipped_mat_geom.items():
            assert len(particles) == len(
                flipped_mat_geom[tomo_id]
            ), f"Number of particles in {tomo_id} is not the same as in the original file. Original: {len(mat_geom[tomo_id])}, Flipped: {len(flipped_mat_geom[tomo_id])}"

        log_test_success(test_name, logger)
        logger.info(f"Flipped data saved to: {output_file}")
        logger.info(
            f"Successfully flipped {flipped_count} particles across {len(particles_to_flip)} tomograms"
        )

    except Exception as e:
        log_test_failure(test_name, e, logger)
        raise


if __name__ == "__main__":
    test_flip_emc_particles()
    print("Test completed successfully!")
