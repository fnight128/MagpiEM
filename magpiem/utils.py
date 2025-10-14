# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""

import logging

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
