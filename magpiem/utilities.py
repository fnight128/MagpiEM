# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""

import numpy as np


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
