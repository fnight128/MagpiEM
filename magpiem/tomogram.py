# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""

import math
import logging
from collections import defaultdict
import numpy as np

from .cleaner import Cleaner
from .particle import Particle
from .utilities import within
from .plotting_helpers import colour_range, create_lattice_plot_from_raw_data

logger = logging.getLogger(__name__)

WHITE = "#FFFFFF"
GREY = "#646464"
BLACK = "#000000"


class Tomogram:
    name: str
    all_particles: set
    selected_lattices: set

    lattices: defaultdict

    lattice_df_dict: dict

    cleaning_params: Cleaner

    __ADJ_AREAS = tuple(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )

    def __init__(self, name):
        self.lattices = defaultdict(lambda: set())
        self.lattices[0] = set()
        self.name = name
        self.selected_lattices = set()

    def __hash__(self):
        """
        Hash the tomogram by its name. Files cannot contain multiple tomograms with the same name, so should be stable
        """
        return self.name

    def __str__(self):
        return "Tomogram {}, containing {} particles. Selected lattices: {}".format(
            self.name,
            len(self.all_particles),
            ",".join({str(n) for n in self.selected_lattices}),
        )

    @staticmethod
    def assign_regions(particles: set["Particle"], max_dist: float) -> dict:
        """
        Assign all particles to regions, to speed up finding neighbours

        Parameters
        ----------
        particles
            Set of all particles in tomogram
        max_dist
            Maximum distance for particles to be considered neighbours
        Returns
        -------
            Dictionary of regions with their particles
        """
        regions = defaultdict(lambda: set())
        for particle in particles:
            position_list = [str(math.floor(q / max_dist)) for q in particle.position]
            locality_id = "_".join(position_list)
            regions[locality_id].add(particle)
        return regions

    @staticmethod
    def find_nearby_keys(region_key: str) -> list[str]:
        """
        Given a region key, find all 27 adjacent keys (including itself)
        Parameters
        ----------
        region_key
            e.g. 4_7_2

        Returns
        -------
            List of regions e.g.
            3_6_1 to 5_8_3
        """
        coords = [int(q) for q in region_key.split("_")]
        return [
            "_".join([str(q) for q in np.array(coords) + np.array(adj_orientation)])
            for adj_orientation in Tomogram.__ADJ_AREAS
        ]

    @staticmethod
    def find_nearby_particles(regions: dict, region_key: str) -> set["Particle"]:
        """
        Get all particles from all 27 nearby regions
        Parameters
        ----------
        regions
            All regions in tomogram
        region_key
            Region to find nearby particles from

        Returns
        -------
            Set of all nearby particles
        """
        return set().union(
            *[regions[k] for k in Tomogram.find_nearby_keys(region_key) if k in regions]
        )

    def reset_cleaning(self) -> None:
        """
        Reset tomogram to initial state
        Deletes all lattices and un-assigns all particles
        """
        keys = set(self.lattices.keys()).copy()
        for n in keys:
            self.delete_lattice(n)
        self.lattices[0] = set()

    def find_particle_neighbours(self, drange=None) -> None:
        """
        Assign neighbours to all particles in tomogram, according to given
            distance range
        Parameters
        ----------
        drange
            Ordered list of min/max distances, squared.
        """
        if not drange:
            drange = self.cleaning_params.dist_range
        particles = self.all_particles
        regions = Tomogram.assign_regions(particles, max(drange) ** 0.5)
        # Start with most populated region, so to maximise the number of particles which
        # do not need to be checked again
        sorted_region_keys = sorted(
            regions, key=lambda k: len(regions[k]), reverse=True
        )

        for r_key in sorted_region_keys:
            if len(regions[r_key]) == 0:
                continue
            proximal_particles = Tomogram.find_nearby_particles(regions, r_key)

            for particle in regions[r_key]:
                for particle2 in proximal_particles:
                    if particle == particle2:
                        continue
                    if within(particle.distance_sq(particle2), drange):
                        particle.make_neighbours(particle2)
            # clear region so no particles are checked twice
            regions[r_key] = set()

    def get_particle_from_position(
        self, position: np.ndarray, precision: int = 0
    ) -> "Particle":
        """
        Get a particle based on its position, to 'precision' decimal places
        If no particle found, returns None

        Parameters
        ----------
        position
            [x,y,z] position of particle
        precision
            Number of decimal places to round position to
        Returns
        -------
            Located particle
        """
        rough_pos = np.around(position, decimals=precision)
        for particle in self.all_particles:
            if all(np.around(particle.position, decimals=precision) == rough_pos):
                return particle

    def assign_particles(self, particles: set["Particle"]) -> None:
        """
        Assign set of particles to tomogram
        """
        self.all_particles = particles

    def set_clean_params(self, cp: "Cleaner") -> None:
        """Assign cleaning parameters to tomogram"""
        self.cleaning_params = cp

    def selected_particle_ids(self, selected: bool = True) -> set[int]:
        """
        Get all manually selected particle ids from tomogram
        If selected is False, instead get unselected particles
        Parameters
        ----------
        selected
            True for selected particles (default)
            False for unselected particles
        Returns
        -------
            set of selected particle ids
        """
        return {
            particle.particle_id
            for particle in self.get_auto_cleaned_particles()
            if (particle.lattice in self.selected_lattices) == selected
        }

    def autoclean(self) -> dict:
        """
        Clean all particles in tomogram according to its
        assigned cleaning params
        Tomogram must first be assigned a cleaner
        Returns dict of lattice ids and their particle ids
            {lattice_id: set[particle_ids...] ...}
        """
        self.all_particles = {
            particle
            for particle in self.all_particles
            if particle.cc_score > self.cleaning_params.cc_threshold
        }

        self.find_particle_neighbours()
        for particle in self.all_particles:
            neighbours = particle.neighbours

        for particle in self.all_particles:
            particle.filter_neighbour_orientation(
                self.cleaning_params.ori_range, self.cleaning_params.flipped_ori_range
            )

        for particle in self.all_particles:
            particle.filter_curvature(self.cleaning_params.curv_range)

        # Process particles in order of ID for deterministic lattice assignment
        # Necessary for reproducible lattice assignment and comparison with C++
        sorted_particles = sorted(self.all_particles, key=lambda p: p.particle_id)
        for particle in sorted_particles:
            if (
                particle.lattice
                or len(particle.neighbours) < self.cleaning_params.min_neighbours
            ):
                continue
            particle.choose_new_lattice(len(self.lattices))

        bad_lattices = set()
        for lattice_key, lattice in self.lattices.items():
            if len(lattice) < self.cleaning_params.min_lattice_size:
                bad_lattices.add(lattice_key)
        for lattice_key in bad_lattices:
            self.delete_lattice(lattice_key)
        return self.write_progress_dict()

    def toggle_selected(self, n: int) -> None:
        """
        Toggle whether lattice n is manually selected or not
        Note that lattice 0 can never be selected - this represents
        unclean particles
        """
        if n in self.selected_lattices:
            self.selected_lattices.remove(n)
        # 0 always represents unclean particles - cannot be selected
        elif n != 0:
            self.selected_lattices.add(n)

    def delete_lattice(self, n: int) -> None:
        """Delete lattice from tomogram and un-assign all its particles"""
        array_particles = self.lattices[n].copy()
        for particle in array_particles:
            particle.set_lattice(0)
        self.lattices.pop(n)

    def write_progress_dict(self) -> dict[str:dict]:
        """
        Generate dict of current progress in tomogram
        Keys are tomogram ids, values are dicts of
        lattice id: set[lattice particles], each also containing
        "selected": set[selected_lattice_ids]

        Returns
        -------
            Dict
        """
        if len(self.lattices.keys()) < 2:
            logger.warning("Skipping tomo {}, contains no lattices".format(self.name))
            return
        arr_dict = {}
        for ind, arr in self.lattices.items():
            # must use lists, or yaml interprets as dicts
            if ind == 0:
                continue
            p_ids = [p.particle_id for p in arr]
            arr_dict[ind] = p_ids
        selected_lattices = list(self.selected_lattices)
        arr_dict["selected"] = selected_lattices
        return arr_dict

    def apply_progress_dict(self, prog_dict: dict) -> None:
        """
        Reapply previous progress to tomogram, assigning all particles
            to lattices and re-selecting lattices
        Parameters
        ----------
        prog_dict
            Dict formatted according to Tomogram.write_progress_dict
        """
        if not prog_dict:
            return

        inverted_prog_dict = {}
        for array, particles in prog_dict.items():
            # skip dict which stores which arrays are selected
            if array == "selected":
                continue
            for particle in particles:
                inverted_prog_dict[particle] = array

        for particle in self.all_particles:
            p_id = particle.particle_id
            if p_id in inverted_prog_dict.keys():
                particle.set_lattice(int(inverted_prog_dict[particle.particle_id]))
            else:
                particle.set_lattice(0)
        self.selected_lattices = set(prog_dict["selected"])

    def plot_all_lattices(self, **kwargs) -> "go.Figure":
        """
        Returns a figure of all lattices in a tomogram using plotting_helpers

        kwargs:
            "showing_removed_particles": True to show unclean particles. Defaults to False
            "cone_size": size of cones to plot. Set to negative value to instead plot points.
            "cone_fix": True to include a small set of cones which forces all cones to have the same
                size. Always recommended. Defaults to True
        """
        # Convert tomogram data to raw format for plotting_helpers
        raw_particle_data = [
            [particle.position.tolist(), particle.orientation.tolist()]
            for particle in self.all_particles
        ]

        # Convert lattice data to the format expected by plotting_helpers
        lattice_data = {}
        for lattice_id, particles in self.lattices.items():
            if lattice_id == 0 and not kwargs.get("showing_removed_particles", False):
                continue
            particle_indices = [
                i
                for i, particle in enumerate(self.all_particles)
                if particle in particles
            ]
            if particle_indices:
                lattice_data[lattice_id] = particle_indices

        # Create plot
        cone_size = kwargs.get("cone_size", 0)
        show_removed = kwargs.get("showing_removed_particles", False)
        selected_lattices = self.selected_lattices

        return create_lattice_plot_from_raw_data(
            raw_particle_data,
            lattice_data,
            cone_size=cone_size,
            show_removed_particles=show_removed,
            selected_lattices=selected_lattices,
        )

    def to_dict(self) -> dict:
        """
        Serialise for JSON conversion
        """
        return {
            "name": self.name,
            "all_particles": [particle.to_dict() for particle in self.all_particles],
        }

    @staticmethod
    def from_dict(tomo_dict: dict) -> "Tomogram":
        """
        Deserialise from JSON dict
        """
        tomo = Tomogram(tomo_dict["name"])
        tomo_particles = {
            Particle.from_dict(particle_dict, tomo)
            for particle_dict in tomo_dict["all_particles"]
        }
        tomo.assign_particles(tomo_particles)
        return tomo
