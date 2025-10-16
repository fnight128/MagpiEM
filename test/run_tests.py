#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test runner for MagpiEM tests using pytest.

This script provides a unified way to run all tests with standardised logging
and configuration. It supports running individual test categories or all tests.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add test utilities to path
test_root = Path(__file__).parent
sys.path.insert(0, str(test_root))

from test_utils import setup_test_logging, setup_test_environment  # noqa: E402


def run_pytest_tests(test_path: str, logger) -> bool:
    """Run pytest on a specific test path."""
    try:
        import logging

        # Get the current log level
        log_level = logging.getLogger().level
        log_level_name = logging.getLevelName(log_level)

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--strict-markers",  # Strict marker checking
            "-s",  # Don't capture output (so print statements show)
            "--log-cli-level",
            log_level_name,  # Set pytest log level
            "--log-cli-format",
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        # Run pytest
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        logger.error(f"Failed to run pytest: {e}")
        return False


def run_unit_tests(logger):
    """Run unit tests."""
    logger.info("Running unit tests...")
    return run_pytest_tests(test_root / "unit", logger)


def run_integration_tests(logger):
    """Run integration tests."""
    logger.info("Running integration tests...")
    return run_pytest_tests(test_root / "integration", logger)


def run_e2e_tests(logger):
    """Run end-to-end tests."""
    logger.info("Running end-to-end tests...")
    return run_pytest_tests(test_root / "e2e", logger)


def run_all_tests(logger):
    """Run all tests."""
    logger.info("Running all tests...")
    return run_pytest_tests(test_root, logger)


def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests")
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "e2e", "all"],
        default="all",
        help="Test category to run (default: all)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--log-file", help="Log file path (optional)")
    parser.add_argument("--pytest-args", help="Additional arguments to pass to pytest")

    args = parser.parse_args()

    # Set up test environment
    setup_test_environment()

    # Set up logging
    import logging

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_test_logging(level=log_level, log_file=args.log_file)

    logger.info(f"Starting test suite - Category: {args.category}")

    # Run tests based on category
    if args.category == "unit":
        success = run_unit_tests(logger)
    elif args.category == "integration":
        success = run_integration_tests(logger)
    elif args.category == "e2e":
        success = run_e2e_tests(logger)
    else:  # all
        success = run_all_tests(logger)

    if success:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
