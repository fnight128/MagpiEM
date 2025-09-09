# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""

import atexit
import glob
import logging
import os
import signal
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)


def normalise(vec: np.ndarray):
    """Normalise vector"""
    assert not all(x == 0 for x in vec), "Attempted to normalise a zero vector"
    mag = np.linalg.norm(vec)
    return vec / mag


def within(value: float, allowed_range: tuple):
    """
    Check whether 'value' is within 'allowed range'

    Parameters
    ----------
    value : float
        Value to check
    allowed_range : tuple
        Ordered tuple consisting of (min_val, max_val)

    Returns
    -------
    bool
        allowed_range[0] <= value <= allowed_range[1]

    """
    return allowed_range[0] <= value <= allowed_range[1]


def clamp(n: float, lower_bound: float, upper_bound: float) -> float:
    """
    Force n to be between 'lower_bound' and 'upper_bound"
    If n within range, return n
    If n < lower_bound, return lower_bound
    If n > upper_bound, return upper_bound
    Parameters
    ----------
    n
    lower_bound
    upper_bound

    Returns
    -------
    Clamped value of n
    """
    return max(min(upper_bound, n), lower_bound)


def clear_cache_directory(cache_dir):
    """Clear all files from the cache directory except cleaning parameters."""
    if not os.path.exists(cache_dir):
        return

    logger.info("Clearing cache directory: %s", cache_dir)

    try:
        cache_files = glob.glob(os.path.join(cache_dir, "*"))

        for file_path in cache_files:
            if os.path.isfile(file_path):
                try:
                    # Skip dash diskcache - dash will handle them itself, and leads to errors if we interfere
                    # cache.db may remain, but not large enough to matter. reset on next run regardless
                    if (
                        file_path.endswith(".db")
                        or file_path.endswith(".db-shm")
                        or file_path.endswith(".db-wal")
                    ):
                        continue

                    os.remove(file_path)
                    logger.debug("Removed cached file: %s", file_path)
                except OSError as e:
                    logger.warning("Could not remove cached file %s: %s", file_path, e)

        logger.info("Cache directory cleared successfully")
    except Exception as e:
        logger.error("Error clearing cache directory: %s", e)


def setup_cleanup_handlers(cache_dir):
    """Set up signal handlers and atexit handlers to clear cache on termination."""
    logger = logging.getLogger(__name__)

    # "frame" necessary for signature
    def cleanup_handler(signum=None, frame=None):
        """Handle cleanup on termination."""
        logger.info("Application terminating, clearing cache...")
        clear_cache_directory(cache_dir)
        # ensure proper termination
        if signum:
            sys.exit(0)

    # run on normal exit
    atexit.register(cleanup_handler)

    # run on various termination signals
    signal.signal(signal.SIGINT, cleanup_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_handler)  # Termination signal

    # on windows, also SIGBREAK
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, cleanup_handler)


def validate_required_data(*data_items):
    """Validate that required data items are not None or empty."""
    return all(item is not None and item != "" for item in data_items)


def log_callback_start(callback_name, **kwargs):
    """Log callback start with relevant parameters."""
    logger = logging.getLogger(__name__)
    logger.debug(
        f"{callback_name} called with: {', '.join(f'{k}={v}' for k, v in kwargs.items())}"
    )


def handle_callback_error(callback_name, error, default_return=None):
    """Handle callback errors with consistent logging."""
    logger = logging.getLogger(__name__)
    logger.error(f"Error in {callback_name}: {error}")
    return default_return
