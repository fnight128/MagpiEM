#!/usr/bin/env python3
"""
Unit tests for magpiem.io.io_utils module.

Tests the core I/O functionality including file reading, writing, and data processing.
"""

import sys
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock

# Add test utilities to path
test_root = Path(__file__).parent.parent
sys.path.insert(0, str(test_root))

from test_utils import (  # noqa: E402
    setup_test_logging,
    log_test_start,
    log_test_success,
    log_test_failure,
    setup_test_environment,
)

setup_test_environment()

from magpiem.io.io_utils import (  # noqa: E402
    purge_blank_tomos,
    flip_emc_particles,
    write_emc_mat,
    clear_cache_directory,
    read_emc_mat,
    read_emc_tomogram,
    read_emc_tomogram_raw_data,
    get_tomogram_names,
    process_uploaded_file,
    load_previous_session,
    X_180_ROTATION_MATRIX,
)
from magpiem.processing.classes.tomogram import Tomogram  # noqa: E402

logger = setup_test_logging()

# Test data file path
TEST_DATA_FILE = test_root / "data" / "test_data.mat"


class TestIOUtils:
    """Test class for io_utils functions."""

    def test_purge_blank_tomos(self):
        """Test purge_blank_tomos function."""
        test_name = "purge_blank_tomos"
        log_test_start(test_name, logger)

        try:
            # Create test data
            mat_dict = {
                "tomo1": {"data": "value1"},
                "tomo2": {"data": "value2"},
                "tomo3": {"data": "value3"},
                "nested": {"tomo1": "nested_value1", "tomo4": "nested_value4"},
            }
            blank_tomos = {"tomo1", "tomo3"}

            # Make a copy for testing
            test_dict = mat_dict.copy()
            test_dict["nested"] = mat_dict["nested"].copy()

            purge_blank_tomos(test_dict, blank_tomos)

            # Check that blank tomos are removed
            assert "tomo1" not in test_dict
            assert "tomo3" not in test_dict
            assert "tomo2" in test_dict  # Should remain

            # Check nested removal
            assert "tomo1" not in test_dict["nested"]
            assert "tomo4" in test_dict["nested"]  # Should remain

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_flip_emc_particles(self):
        """Test flip_emc_particles function."""
        test_name = "flip_emc_particles"
        log_test_start(test_name, logger)

        try:
            # Create test geometry data
            mat_geom = {
                "tomo1": {
                    0: [0] * 16
                    + [1, 0, 0, 0, 1, 0, 0, 0, 1]
                    + [0] * 6,  # Identity matrix
                    1: [0] * 16 + [1, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 6,
                }
            }
            particles_to_flip = {"tomo1": [0]}

            result = flip_emc_particles(mat_geom, particles_to_flip)

            # Check that the rotation matrix was modified
            original_matrix = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)
            expected_matrix = X_180_ROTATION_MATRIX @ original_matrix

            flipped_matrix = np.array(result["tomo1"][0][16:25]).reshape(3, 3)
            np.testing.assert_array_almost_equal(flipped_matrix, expected_matrix)

            # Check that particle 1 was not flipped
            unchanged_matrix = np.array(result["tomo1"][1][16:25]).reshape(3, 3)
            np.testing.assert_array_almost_equal(unchanged_matrix, original_matrix)

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_clear_cache_directory(self):
        """Test clear_cache_directory function."""
        test_name = "clear_cache_directory"
        log_test_start(test_name, logger)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create some test files
                test_files = ["test1.txt", "test2.txt", "cache.db", "cache.db-shm"]
                for filename in test_files:
                    with open(os.path.join(temp_dir, filename), "w") as f:
                        f.write("test content")

                # Call clear_cache_directory
                clear_cache_directory(temp_dir)

                # Check that non-db files are removed
                assert not os.path.exists(os.path.join(temp_dir, "test1.txt"))
                assert not os.path.exists(os.path.join(temp_dir, "test2.txt"))

                # Check that db files are preserved
                assert os.path.exists(os.path.join(temp_dir, "cache.db"))
                assert os.path.exists(os.path.join(temp_dir, "cache.db-shm"))

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_read_emc_mat(self):
        """Test read_emc_mat function."""
        test_name = "read_emc_mat"
        log_test_start(test_name, logger)

        try:
            if not TEST_DATA_FILE.exists():
                logger.warning(
                    f"Test data file {TEST_DATA_FILE} not found, skipping test"
                )
                return

            result = read_emc_mat(str(TEST_DATA_FILE))

            # Check that result is a dictionary
            assert isinstance(result, dict)
            assert len(result) > 0

            # Check that values are lists (particle data)
            for tomo_name, particles in result.items():
                assert isinstance(particles, list)
                if len(particles) > 0:
                    # Check that each particle has the expected structure
                    particle = particles[0]
                    assert isinstance(particle, list)
                    assert len(particle) >= 25  # Expected minimum length

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_read_emc_tomogram(self):
        """Test read_emc_tomogram function."""
        test_name = "read_emc_tomogram"
        log_test_start(test_name, logger)

        try:
            if not TEST_DATA_FILE.exists():
                logger.warning(
                    f"Test data file {TEST_DATA_FILE} not found, skipping test"
                )
                return

            geom_data = read_emc_mat(str(TEST_DATA_FILE))
            if not geom_data:
                logger.warning("No geometry data found, skipping test")
                return

            tomo_name = list(geom_data.keys())[0]
            tomo_geometry = geom_data[tomo_name]

            result = read_emc_tomogram(tomo_geometry, tomo_name)

            assert isinstance(result, Tomogram)
            assert result.name == tomo_name
            assert len(result.particles) > 0

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_read_emc_tomogram_raw_data(self):
        """Test read_emc_tomogram_raw_data function."""
        test_name = "read_emc_tomogram_raw_data"
        log_test_start(test_name, logger)

        try:
            if not TEST_DATA_FILE.exists():
                logger.warning(
                    f"Test data file {TEST_DATA_FILE} not found, skipping test"
                )
                return

            geom_data = read_emc_mat(str(TEST_DATA_FILE))
            if not geom_data:
                logger.warning("No geometry data found, skipping test")
                return

            tomo_name = list(geom_data.keys())[0]
            tomo_geometry = geom_data[tomo_name]

            result = read_emc_tomogram_raw_data(tomo_geometry, tomo_name)

            assert isinstance(result, list)
            assert len(result) > 0

            if len(result) > 0:
                particle = result[0]
                assert isinstance(particle, list)
                assert len(particle) == 2  # [position, orientation]
                assert len(particle[0]) == 3  # position
                assert len(particle[1]) == 3  # orientation

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_get_tomogram_names(self):
        """Test get_tomogram_names function."""
        test_name = "get_tomogram_names"
        log_test_start(test_name, logger)

        try:
            if not TEST_DATA_FILE.exists():
                logger.warning(
                    f"Test data file {TEST_DATA_FILE} not found, skipping test"
                )
                return

            result = get_tomogram_names(str(TEST_DATA_FILE))

            assert isinstance(result, list)
            assert len(result) > 0
            for name in result:
                assert isinstance(name, str)

            limited_result = get_tomogram_names(str(TEST_DATA_FILE), num_images=2)
            assert len(limited_result) <= 2

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_write_emc_mat(self):
        """Test write_emc_mat function."""
        test_name = "write_emc_mat_integration"
        log_test_start(test_name, logger)

        try:
            if not TEST_DATA_FILE.exists():
                logger.warning(
                    f"Test data file {TEST_DATA_FILE} not found, skipping test"
                )
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                keep_ids = {"wt2nd_4004_2": [0, 1, 2]}  # Keep first 3 particles

                output_path = os.path.join(temp_dir, "test_output.mat")

                write_emc_mat(
                    keep_ids, output_path, str(TEST_DATA_FILE), purge_blanks=True
                )

                assert os.path.exists(output_path)

                output_data = read_emc_mat(output_path)
                assert isinstance(output_data, dict)
                assert "wt2nd_4004_2" in output_data
                assert len(output_data["wt2nd_4004_2"]) == 3

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_process_uploaded_file(self):
        """Test process_uploaded_file function."""
        test_name = "process_uploaded_file"
        log_test_start(test_name, logger)

        try:
            mock_save_dash_upload = Mock()
            mock_get_tomogram_names = Mock(return_value=["tomo1", "tomo2"])

            result, error = process_uploaded_file(
                "test.mat",
                b"mock_content",
                0,  # num_images
                "/tmp/",
                mock_save_dash_upload,
                mock_get_tomogram_names,
            )

            assert result is not None
            assert error is None
            assert "__tomogram_names__" in result
            assert result["__tomogram_names__"] == ["tomo1", "tomo2"]

            # Test with no filename
            result, error = process_uploaded_file(
                None,
                b"mock_content",
                0,
                "/tmp/",
                mock_save_dash_upload,
                mock_get_tomogram_names,
            )

            assert result is None
            assert "Please choose a particle database" in error

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise

    def test_load_previous_session(self):
        """Test load_previous_session function."""
        test_name = "load_previous_session"
        log_test_start(test_name, logger)

        try:
            mock_ctx = Mock()
            mock_ctx.triggered_id = "upload-previous-session"

            mock_save_dash_upload = Mock()
            mock_read_previous_progress = Mock(return_value={"test": "data"})

            result, error = load_previous_session(
                "test.json",
                b"mock_content",
                {"existing": "data"},
                "/tmp/",
                mock_ctx,
                mock_save_dash_upload,
                mock_read_previous_progress,
            )

            assert result == {"test": "data"}
            assert error is None

            mock_ctx.triggered_id = "other-trigger"
            result, error = load_previous_session(
                "test.json",
                b"mock_content",
                {"existing": "data"},
                "/tmp/",
                mock_ctx,
                mock_save_dash_upload,
                mock_read_previous_progress,
            )

            assert result == {}
            assert error == {}

            log_test_success(test_name, logger)

        except Exception as e:
            log_test_failure(test_name, e, logger)
            raise


def run_all_tests():
    """Run all tests in this module."""
    test_instance = TestIOUtils()
    test_methods = [
        method for method in dir(test_instance) if method.startswith("test_")
    ]

    logger.info(f"Running {len(test_methods)} tests for io_utils module")

    for test_method in test_methods:
        try:
            getattr(test_instance, test_method)()
        except Exception as e:
            logger.error(f"Test {test_method} failed: {e}")
            continue

    logger.info("All io_utils tests completed")


if __name__ == "__main__":
    run_all_tests()
