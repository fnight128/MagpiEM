# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""

import numpy as np
from typing import Tuple, Any
from .utilities import within


class Cleaner:
    cc_threshold: float
    min_neighbours: int
    min_lattice_size: int
    dist_range: tuple
    ori_range: tuple
    curv_range: tuple
    flipped_ori_range: tuple

    def __init__(
        self,
        cc_thresh: float,
        min_neighbours: int,
        min_lattice_size: int,
        dist_range: tuple[float],
        ori_range: tuple[float],
        curv_range: tuple[float],
        flipped_ori_range: tuple[float] | None,
    ):
        self.cc_threshold = cc_thresh
        self.min_neighbours = min_neighbours
        self.min_lattice_size = min_lattice_size
        self.dist_range = dist_range
        self.ori_range = ori_range
        self.curv_range = curv_range
        self.flipped_ori_range = flipped_ori_range

    @staticmethod
    def from_user_params(
        cc_thresh: float,
        min_neigh: int,
        min_lattice_size: int,
        target_dist: float,
        dist_tol: float,
        target_ori: float,
        ori_tol: float,
        target_curv: float,
        curv_tol: float,
        allow_flips: bool = False,
    ) -> "Cleaner":
        """
        Define a set of cleaning parameters from user specification
        """
        dist_range = Cleaner.dist_range(target_dist, dist_tol)
        ori_range = Cleaner.ang_range_dotprod(target_ori, ori_tol)
        curv_range = Cleaner.ang_range_dotprod(target_curv, curv_tol)
        flipped_ori_range = (-x for x in reversed(ori_range)) if allow_flips else 0
        return Cleaner(
            cc_thresh,
            min_neigh,
            min_lattice_size,
            dist_range,
            ori_range,
            curv_range,
            flipped_ori_range,
        )

    def to_dict(self) -> dict:
        """
        Serialise
        """
        return {
            "cc threshold": self.cc_threshold,
            "min neighbours": self.min_neighbours,
            "min lattice size": self.min_lattice_size,
            "distance range": self.dist_range,
            "orientation range": self.ori_range,
            "curvature range": self.curv_range,
            "flipped ori range": self.flipped_ori_range,
        }

    @staticmethod
    def from_dict(clean_dict: dict) -> "Cleaner":
        """
        Deserialise
        """
        return Cleaner(**clean_dict)

    def __str__(self):
        dist_range = [np.round(dist**0.5, decimals=1) for dist in self.dist_range]
        angle_ranges = [
            np.round(np.degrees(np.arccos(ang)), decimals=1)
            for ang in [*self.ori_range, *self.curv_range]
        ]
        return "Allowed distances: {}-{}. Allowed orientations:{}-{}. Allowed curvatures:{}-{}.".format(
            *dist_range, *angle_ranges
        )

    @staticmethod
    def dist_range(target_dist: float, dist_tol: float) -> tuple[float, float]:
        """
        Create an ordered list of distances squared from 'target_dist' ± 'dist_tol'
        If lower bound would be < 0, it is clamped to 0.
        Parameters
        ----------
        target_dist
        dist_tol

        Returns
        -------
        Ordered list
        """
        dist_tol = abs(dist_tol)
        assert target_dist != 0, "Target distance cannot be 0"
        if target_dist < 0:
            target_dist = abs(target_dist)
            print("Target distance must be > 0, correcting to ", target_dist)
        return (
            (
                (target_dist - dist_tol) ** 2
                if dist_tol < target_dist
                else 0.0001 * dist_tol
            ),
            (target_dist + dist_tol) ** 2,
        )

    @staticmethod
    def ang_range_dotprod(
        angle_ideal: float, angle_tolerance: float
    ) -> tuple[float, ...]:
        """
        Given an angle and tolerance, generate an ordered list representing the
        dot product of unit vectors at these angles

        Parameters
        ----------
        angle_ideal : float
            Desired angle in degrees
        angle_tolerance : float
            Tolerance in degrees

        Returns
        -------
        Ordered list of dot products
        """

        if not within(angle_ideal, (0, 180)):
            angle_ideal = angle_ideal % 180
            print("Angle between adjacent particles must be between 0 and 180 degrees")
            print("Corrected angle: ", angle_ideal)
        elif not within(angle_tolerance, (0, 180)):
            angle_tolerance = angle_tolerance % 180
            print("Angle tolerance must be between 0 and 180 degrees")
            print("Corrected tolerance: ", angle_tolerance)
        min_ang = angle_ideal - angle_tolerance
        max_ang = angle_ideal + angle_tolerance

        # edge cases where tolerance extends beyond [0,180]
        if min_ang < 0:
            max_ang = max(max_ang, -min_ang)
            min_ang = 0
        elif max_ang > 180:
            min_ang = min(min_ang, 360 - max_ang)
            max_ang = 180

        return tuple([float(np.cos(np.radians(ang))) for ang in [max_ang, min_ang]])
