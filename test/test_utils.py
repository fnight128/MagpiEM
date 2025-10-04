# -*- coding: utf-8 -*-
"""
Common utilities for MagpiEM tests.

This module provides standardised logging, configuration, and common test functions
that can be used across all test files to ensure consistency.
"""

import logging
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

# Suppress common warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TestConfig:
    """Configuration class for test settings."""

    # Test data files
    TEST_DATA_MINISCULE = "test_data_miniscule.mat"
    TEST_DATA_STANDARD = "test_data.mat"
    TEST_DATA_LARGE = "WT_CA_2nd.mat"

    # Test tomogram names
    TEST_TOMO_MINISCULE = "test_tomo"
    TEST_TOMO_STANDARD = "wt2nd_4004_2"
    TEST_TOMO_LARGE = "wt2nd_4004_6"

    # Standard test parameters for cleaning
    TEST_CLEANER_VALUES = [2.0, 3, 10, 60.0, 40.0, 10.0, 20.0, 90.0, 20.0]

    # Dash UI test parameters
    TEST_DASH_PARAMETERS = {
        "switch-allow-flips": False,
        "inp-cc-thresh": 3,
        "inp-curv-goal": 90,
        "inp-curv-tol": 20,
        "inp-dist-goal": 60,
        "inp-dist-tol": 20,
        "inp-array-size": 3,
        "inp-min-neighbours": 2,
        "inp-ori-goal": 10,
        "inp-ori-tol": 20,
    }


def setup_test_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up standardised logging for tests.

    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)
    log_file : str, optional
        Path to log file. If None, logs only to console
    format_string : str, optional
        Custom format string. If None, uses standard format

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s %(name)s %(levelname)s: %(message)s"

    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            (
                logging.FileHandler(log_file, mode="w")
                if log_file
                else logging.NullHandler()
            ),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Test logging configured - Level: {logging.getLevelName(level)}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    return logger


def get_test_data_path(filename: str) -> Path:
    """
    Get the full path to a test data file.

    Parameters
    ----------
    filename : str
        Name of the test data file

    Returns
    -------
    Path
        Full path to the test data file
    """
    test_root = Path(__file__).parent
    data_path = test_root / "data" / filename

    if not data_path.exists():
        # Fallback to old location for backward compatibility
        fallback_path = test_root / filename
        if fallback_path.exists():
            return fallback_path
        else:
            raise FileNotFoundError(f"Test data file not found: {filename}")

    return data_path


def create_temp_test_dir() -> Path:
    """
    Create a temporary directory for test files.

    Returns
    -------
    Path
        Path to the created temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="magpiem_test_"))
    return temp_dir


def cleanup_temp_dir(temp_dir: Path) -> None:
    """
    Clean up a temporary test directory.

    Parameters
    ----------
    temp_dir : Path
        Path to the temporary directory to clean up
    """
    import shutil

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns
    -------
    Path
        Path to the project root
    """
    return Path(__file__).parent.parent


def setup_test_environment() -> Dict[str, Any]:
    """
    Set up the test environment with common configuration.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing test environment configuration
    """
    project_root = get_project_root()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    magpiem_path = project_root / "magpiem"
    if str(magpiem_path) not in sys.path:
        sys.path.insert(0, str(magpiem_path))

    logs_dir = project_root / "test" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "logs_dir": logs_dir,
        "test_data_dir": project_root / "test" / "data",
    }


# Common test decorators and utilities
def skip_if_no_data(filename: str):
    """
    Decorator to skip tests if required test data is not available.

    Parameters
    ----------
    filename : str
        Name of the test data file required
    """

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            try:
                data_path = get_test_data_path(filename)
                if not data_path.exists():
                    return unittest.skip(f"Test data not available: {filename}")(
                        test_func
                    )(*args, **kwargs)
                return test_func(*args, **kwargs)
            except FileNotFoundError:
                return unittest.skip(f"Test data not available: {filename}")(test_func)(
                    *args, **kwargs
                )

        return wrapper

    return decorator


def log_test_start(test_name: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log the start of a test with a standard format.

    Parameters
    ----------
    test_name : str
        Name of the test being started
    logger : logging.Logger, optional
        Logger to use. If None, uses the root logger
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"Starting test: {test_name}")


def log_test_success(test_name: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log the successful completion of a test.

    Parameters
    ----------
    test_name : str
        Name of the test that completed
    logger : logging.Logger, optional
        Logger to use. If None, uses the root logger
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"PASSED Test: {test_name}")


def log_test_failure(
    test_name: str, error: Exception, logger: Optional[logging.Logger] = None
) -> None:
    """
    Log the failure of a test.

    Parameters
    ----------
    test_name : str
        Name of the test that failed
    error : Exception
        The error that caused the test to fail
    logger : logging.Logger, optional
        Logger to use. If None, uses the root logger
    """
    if logger is None:
        logger = logging.getLogger()

    logger.error(f"FAILED Test: {test_name} - {error}")
