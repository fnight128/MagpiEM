# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""

import numpy as np
from typing import Tuple, Any
from ...utils import normalise, within, clamp


class Particle:
    """
    Representation of a particle in a tomogram

    Attributes
    ----------
    particle_id: int
        Unique identifier for this particle. May be chosen
        explicitly or assigned incrementally.
    cc_score : float
        quality score assigned to particle by
        picking software
    position : np array
        [x,y,z] coordinates of particle in pixels
    orientation: np array
        [u,v,w] unit vector describing z-orientation of particle
    tomo: Tomogram
        'Tomogram particle is assigned to
    lattice: set[Particle]
        Protein lattice particle is assigned to
    neighbours: set[Particle]
        Set of particles immediately adjacent to particle
    """

    particle_id: int
    cc_score: float
    position: np.ndarray
    orientation: np.ndarray

    tomo: "Tomogram"
    lattice: int = 0
    neighbours: set["Particle"]

    def __init__(self, p_id, cc, position, orientation, tomo):
        self.particle_id = p_id
        self.cc_score = cc
        self.position = position
        self.orientation = normalise(orientation)
        self.tomo = tomo
        self.neighbours = set()

    def __hash__(self):
        return int.from_bytes(str(self.position).encode("utf-8"), "little")

    def __str__(self):
        return "x:{:.2f}, y:{:.2f}, z:{:.2f}".format(*self.position)

    def output_dict(self):
        """
        Dictionary of particle properties

        Returns
        -------
        Dict of properties about particle:
            "cc": cc score
            "pos": position
            "ori": orientation
        """
        return {"cc": self.cc_score, "pos": self.position, "ori": self.orientation}

    def displacement_from(self, particle: "Particle") -> np.ndarray:
        """
        Displacement vector between particles
        """
        return self.position - particle.position

    def distance_sq(self, particle: "Particle") -> float:
        """
        Squared distance between particles
        """
        disp = self.displacement_from(particle)
        return float(np.vdot(disp, disp))

    def filter_neighbour_orientation(self, orange: tuple, flipped_range: tuple) -> None:
        """
        Remove particles from neighbours if orientation is not within orange

        Parameters
        ----------
        orange:
            Range of orientation dot products
            (ordered list
        flipped_range
            Optional
            Flipped range of orientations if necessary

        Returns
        -------
        None
        """
        good_orientation = set()
        for neighbour in self.neighbours:
            ori = self.dot_orientation(neighbour)
            if within(ori, orange):
                good_orientation.add(neighbour)
            elif flipped_range and within(ori, flipped_range):
                good_orientation.add(neighbour)
        self.neighbours = good_orientation

    def filter_curvature(self, curv_range) -> None:
        """
        Remove particles from neighbours if curvature is
        incorrect

        Parameters
        ----------
        curv_range
            Ordered list of min and max dot products of curvatures
        Returns
        -------
            None
        """
        good_curvature = {
            neighbour
            for neighbour in self.neighbours
            if within(self.dot_curvature(neighbour), curv_range)
        }
        self.neighbours = good_curvature

    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Dot product of two vectors. Clamped between -1 and 1.
        Parameters
        ----------
        v1, v2
        Vectors to dot product

        Returns
        -------
        Dot product
        """
        dot = float(np.vdot(v1, v2))
        return clamp(dot, -1, 1)

    def dot_orientation(self, particle) -> float:
        """Dot product of two particles' orientations"""
        return Particle.dot_product(self.orientation, particle.orientation)

    def dot_curvature(self, particle) -> float:
        """Dot product of particle's orientation with its displacement from second particle"""
        return Particle.dot_product(
            particle.orientation, normalise(self.displacement_from(particle))
        )

    def choose_new_lattice(self, lattice) -> None:
        """Recursively assign particle and all neighbours to lattice"""
        if len(self.neighbours) < self.tomo.cleaning_params.min_neighbours:
            return
        self.set_lattice(lattice)
        for neighbour in self.neighbours:
            if not neighbour.lattice:
                neighbour.choose_new_lattice(lattice)

    def set_lattice(self, ar: int) -> None:
        """Assign lattice to particle"""
        if self.lattice:
            self.tomo.lattices[self.lattice].discard(self)
        self.lattice = ar
        self.tomo.lattices[ar].add(self)

    def assimilate_lattices(self, assimilate_lattices: set) -> None:
        """
        Combine a set of lattices into a single larger lattice.
        The lattice which all are assimilated into is chosen arbitrarily.

        Parameters
        ----------
        assimilate_lattices : set
            lattices to assimilate

        Returns
        -------
        None.

        """
        # choose a random lattice to assimilate the rest into
        all_lattices = self.tomo.lattices
        particles = self.tomo.all_particles
        assimilate_to = assimilate_lattices.pop()
        for particle in particles:
            if particle.lattice in assimilate_lattices:
                particle.set_lattice(assimilate_to)
        for lattice in assimilate_lattices:
            del all_lattices[lattice]

    def make_neighbours(self, particle2: "Particle") -> None:
        """Define two particles as neighbours"""
        if self is particle2:
            return
        self.neighbours.add(particle2)
        particle2.neighbours.add(self)

    def calculate_params(self, particle2: "Particle") -> dict:
        """Return string describing parameters about two particles"""
        distance = self.distance_sq(particle2) ** 0.5
        orientation = np.degrees(np.arccos(self.dot_orientation(particle2)))
        curvature = np.degrees(np.arccos(self.dot_curvature(particle2)))
        return {
            "Distance": distance,
            "Orientation": orientation,
            "Curvature": curvature,
        }

    @staticmethod
    def from_array(
        plist: list[list], tomo: "Tomogram", ids: list[int] = None
    ) -> set["Particle"]:
        """
        Produce a set of particles from parameters

        Parameters
        ----------
        plist : List of lists of parameters
            List of particles. Each entry in the list
            should be a list of parameters in the
            following order:
                cc value, [x, y, z], [u, v, w]
        tomo: Tomogram
            tomogram object to assign particles to
        ids:
            optional specification of each particle's id
            internally. If not specified, assigned
            incrementally from 0

        Returns
        -------
        Set of particles

        """
        if ids is None:
            return {Particle(idx, *pdata, tomo) for idx, pdata in enumerate(plist)}
        else:
            return {Particle(ids[idx], *pdata, tomo) for idx, pdata in enumerate(plist)}

    def get_avg_curvature(self) -> float:
        """
        Average particle's curvature with all its neighbours
        If has no neighbours, return 0.0
        Returns
        -------
        Average curvature
        """
        if len(self.neighbours) == 0:
            return 0.0
        return float(
            np.mean([self.dot_curvature(neighbour) for neighbour in self.neighbours])
        )

    @staticmethod
    def get_property_array(particles: list["Particle"], prop: str) -> np.ndarray:
        """

        Parameters
        ----------
        particles
            List of N particles from which to extract "prop"
        prop
            Name of property to extract
            Should be either "position" or "orientation"
        Returns
        -------
            Nx3 array specifying property for all particles
        """
        return np.array(
            [getattr(particle, prop) for particle in particles], dtype=float
        )

    def get_neighbour_array(self, prop: str) -> np.ndarray:
        return Particle.get_property_array(self.neighbours, prop)

    def find_flipped_neighbours(self) -> None:
        """
        Recursively assign all neighbours a direction based on their relative orientation
        to the lattice as a whole.
        Requires all neighbours to have been assigned.

        Result will be stored in each particle's "direction" attribute
        """
        for neighbour in self.neighbours:
            if hasattr(neighbour, "direction") and neighbour.direction is not None:
                continue
            if within(
                self.dot_orientation(neighbour), self.tomo.cleaning_params.ori_range
            ):
                neighbour.direction = self.direction
            else:
                neighbour.direction = -self.direction
            neighbour.find_flipped_neighbours()

    def to_dict(self) -> dict:
        """
        Serialise for JSON conversion
        """
        return {
            "particle_id": self.particle_id,
            "cc_score": self.cc_score,
            "position": tuple(self.position),
            "orientation": tuple(self.orientation),
            "lattice": self.lattice,
            # "tomogram" and "neighbours" fields do not need to be stored. Can be reassigned when reading if necessary.
        }

    @staticmethod
    def from_dict(particle_dict: dict, tomogram: "Tomogram") -> "Particle":
        """
        Deserialise from JSON dict
        """
        particle = Particle(
            particle_dict["particle_id"],
            particle_dict["cc_score"],
            np.array(particle_dict["position"]),
            np.array(particle_dict["orientation"]),
            tomogram,
        )
        particle.set_lattice(particle_dict["lattice"])
        return particle
