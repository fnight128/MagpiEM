# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""
import colorsys
from typing import Tuple, Any

import numpy as np
import pandas as pd
import math
from collections import defaultdict
import plotly.graph_objects as go

WHITE = "#FFFFFF"
GREY = "#646464"
BLACK = "#000000"


class Cleaner:
    cc_threshold: float
    min_neighbours: int
    min_lattice_size: int
    dist_range: tuple
    ori_range: tuple
    curv_range: tuple

    flipped_ori_range: list

    dict_to_print: dict

    def __init__(
        self,
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
    ):
        self.cc_threshold = cc_thresh
        self.min_neighbours = min_neigh
        self.min_lattice_size = min_lattice_size
        self.dist_range = Cleaner.dist_range(target_dist, dist_tol)
        self.ori_range = Cleaner.ang_range_dotprod(target_ori, ori_tol)
        self.curv_range = Cleaner.ang_range_dotprod(target_curv, curv_tol)
        self.flipped_ori_range = (
            [-x for x in reversed(self.ori_range)] if allow_flips else 0
        )
        self.dict_to_print = {
            "distance": target_dist,
            "distance tolerance": dist_tol,
            "orientation": target_ori,
            "orientation tolerance": ori_tol,
            "curvature": target_curv,
            "curvature tolerance": curv_tol,
            "cc threshold": cc_thresh,
            "min neighbours": min_neigh,
            "min array size": min_lattice_size,
            "allow flips": allow_flips,
        }

    def __str__(self):
        return "Allowed distances: {}-{}. Allowed orientations:{}-{}. Allowed curvatures:{}-{}.".format(
            *self.dist_range, *self.ori_range, *self.curv_range
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
        self.set_lattice(lattice)
        for neighbour in self.neighbours:
            if not neighbour.lattice:
                neighbour.choose_new_lattice(lattice)

    def set_lattice(self, ar: int) -> None:
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

    def calculate_params(self, particle2: "Particle") -> str:
        """Return string describing parameters about two particles"""
        distance = self.distance_sq(particle2) ** 0.5
        orientation = np.degrees(np.arccos(self.dot_orientation(particle2)))
        curvature = np.degrees(np.arccos(self.dot_curvature(particle2)))
        return "Distance: {:.1f}\nOrientation: {:.1f}°\nDisplacement: {:.1f}°".format(
            distance, orientation, curvature
        )

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


class Tomogram:
    name: str
    all_particles: set
    removed_particles: set
    selected_lattices: set

    checking_particles: list

    lattices: defaultdict

    particles_fate: defaultdict

    lattice_df_dict: dict
    cone_fix: pd.DataFrame

    cleaning_params: Cleaner

    __ADJ_AREAS = tuple(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )

    def __init__(self, name):
        self.lattices = defaultdict(lambda: set())
        self.lattices[0] = set()
        self.name = name
        self.selected_lattices = set()
        self.particles_fate = defaultdict(lambda: set())
        self.checking_particles = []

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

    def find_particle_neighbours_vectorised(self) -> None:
        """
        Assign neighbours to all particles in tomogram, according to given
            distance range.
            Vectorised version of find_particle_neighbours
            Performance increase not significant (~5%),
        """
        print("vec")
        drange = self.cleaning_params.dist_range
        particles = self.all_particles
        regions = Tomogram.assign_regions(particles, max(drange) ** 0.5)
        for r_key, region in regions.items():
            region_particles = list(region)
            if len(region) == 0:
                continue
            proximal_particles = list(Tomogram.find_nearby_particles(regions, r_key))
            proximal_positions = Particle.get_property_array(
                proximal_particles, "position"
            )
            particle_positions = Particle.get_property_array(region, "position")

            proximal_count = len(proximal_particles)
            region_count = len(region_particles)

            proximal_positions_repeated = np.tile(proximal_positions, (region_count, 1))
            particle_positions_repeated = np.tile(
                particle_positions, (proximal_count, 1)
            )
            particle_displacements = (
                proximal_positions_repeated - particle_positions_repeated
            )
            particle_distances_sq = np.einsum(
                "ij,ij->i", particle_displacements, particle_displacements
            )

            for idx, dist_sq in enumerate(particle_distances_sq):
                if within(dist_sq, drange):
                    particle1_index = math.floor(idx / proximal_count)
                    particle2_index = idx % proximal_count
                    region_particles[particle1_index].make_neighbours(
                        proximal_particles[particle2_index]
                    )

            regions[r_key] = set()

    def __hash__(self):
        return self.name

    def __str__(self):
        return "Tomogram {}, containing {} particles. Selected lattices: {}".format(
            self.name,
            len(self.all_particles),
            ",".join({str(n) for n in self.selected_lattices}),
        )

    def get_particle_from_position(self, position: np.ndarray) -> "Particle":
        """
        Get a particle based on its position, to 0 decimal places
        If no particle found, returns None

        Parameters
        ----------
        position
            [x,y,z] position of particle
        Returns
        -------
            Located particle
        """
        rough_pos = np.around(position, decimals=0)
        for particle in self.all_particles:
            if all(np.around(particle.position, decimals=0) == rough_pos):
                return particle

    def get_auto_cleaned_particles(self) -> set["Particle"]:
        """
        Get set of all particles in tomogram assigned as clean by automatic cleaning
        """
        unclean_particles = self.lattices[0]
        return self.all_particles.difference(unclean_particles)

    def assign_particles(self, particles: set["Particle"]) -> None:
        """
        Assign set of particles to tomogram
        """
        self.all_particles = particles
        self.generate_lattice_dfs()
        self.assign_cone_fix_df()

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

    @staticmethod
    def particles_to_df(particles: set["Particle"]) -> pd.DataFrame:
        """
        Create formatted dataframe of particles
        Dataframe has columns:
            "x", "y", "z" for position
            "u", "v", "w" for orientation
            "n" for lattice id

        Parameters
        ----------
        particles
            Particles for df
        Returns
        -------
            df
        """
        particle_data = [
            [*particle.position, *particle.orientation, particle.lattice]
            for particle in particles
        ]
        return pd.DataFrame(particle_data, columns=["x", "y", "z", "u", "v", "w", "n"])

    def all_particles_df(self) -> pd.DataFrame:
        """
        Dataframe of all particles in tomogram

        Formatted as described by :func:`~MagpiEM.Tomogram.particles_to_df`
        """
        return Tomogram.particles_to_df(self.all_particles)

    def checking_particles_df(self) -> pd.DataFrame:
        """
        Dataframe of specific "checking particles" used in dash to
        calculate parameters between two particles

        df formatted as described by :func:`~MagpiEM.Tomogram.particles_to_df`

        Returns
        -------
            df of 0-2 particles
        """
        return Tomogram.particles_to_df(set(self.checking_particles))

    def autoclean(self) -> None:
        """
        Clean all particles in tomogram according to its
        assigned cleaning params
        """
        # TODO simplify this function
        self.all_particles = {
            particle
            for particle in self.all_particles
            if particle.cc_score > self.cleaning_params.cc_threshold
        }

        self.find_particle_neighbours()
        for particle in self.all_particles:
            neighbours = particle.neighbours

            if len(neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["low_neighbours"].add(particle)
                continue
        for particle in self.all_particles:
            particle.filter_neighbour_orientation(
                self.cleaning_params.ori_range, self.cleaning_params.flipped_ori_range
            )
            if len(particle.neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["wrong_ori"].add(particle)
                continue
        for particle in self.all_particles:
            particle.filter_curvature(self.cleaning_params.curv_range)
            if len(particle.neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["wrong_disp"].add(particle)
                continue
        for particle in self.all_particles:
            if particle.lattice:
                continue
            particle.choose_new_lattice(len(self.lattices))

        bad_lattices = set()
        for lattice_key, lattice in self.lattices.items():
            if len(lattice) < self.cleaning_params.min_lattice_size:
                bad_lattices.add(lattice_key)
                for particle in lattice:
                    self.particles_fate["small_array"].add(particle)
        for lattice_key in bad_lattices:
            self.delete_lattice(lattice_key)
        self.generate_lattice_dfs()

    def assign_cone_fix_df(self) -> None:
        """
        Generate a df to fix cone plot size for this tomogram
        See cone_fix_readme.txt for full explanation
        """

        particle_df = self.all_particles_df()

        # locate lower corner of plot, and find overall size
        max_series = particle_df.max()
        min_series = particle_df.min()

        range_series = max_series.subtract(min_series)
        range_magnitude = math.hypot(
            range_series["x"], range_series["y"], range_series["z"]
        )

        # create two extremely small vectors extremely close together
        scaling = 1000000
        fudge_factor = 1000

        min_plus_mag = min_series.add(range_magnitude / (scaling * fudge_factor))

        cone_fix_df = pd.DataFrame(
            [min_series.values, min_plus_mag.values],
            columns=["x", "y", "z", "u", "v", "w", "n"],
        )

        orient = [1 / ((3**0.5) * scaling)] * 2
        for ind in ["u", "v", "w", "n"]:
            cone_fix_df[ind] = orient

        self.cone_fix = cone_fix_df

    def generate_lattice_dfs(self) -> None:
        """
        Generate a dict containing a df of each lattice's particles
        dict keys are lattice ids

        df formatted as described by :func:`~MagpiEM.Tomogram.particles_to_df`
        """
        particle_df = self.all_particles_df()
        # split by lattice
        self.lattice_df_dict = dict(iter(particle_df.groupby("n")))

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

    def get_convex_arrays(self) -> set[int]:
        """Get set of all array ids with an average convex curvature"""
        return {
            a_id
            for a_id, particles in self.lattices.items()
            if np.mean([particle.get_avg_curvature() for particle in particles]) < 0
        }

    def get_concave_arrays(self) -> set[int]:
        """Get set of all array ids with an average concave curvature"""
        # TODO merge with convex
        return set(self.lattices.keys()).difference(self.get_convex_arrays())

    def toggle_convex_arrays(self) -> None:
        """Toggle whether convex arrays are manually selected or not"""
        for a_id in self.get_convex_arrays():
            self.toggle_selected(a_id)

    def toggle_concave_arrays(self) -> None:
        """Toggle whether concave arrays are manually selected or not"""
        # TODO merge with convex
        for a_id in self.get_concave_arrays():
            self.toggle_selected(a_id)

    def show_particle_data(self, position: np.ndarray) -> str:
        """
        Generate human-readable string describing relation between
        two particles
        First call of function assigns one particle
        Second call calculates relation between the two (and resets state)
        Parameters
        ----------
        position
            Position of desired particle

        Returns
        -------
            String describing particles' relationship
        """
        new_particle = self.get_particle_from_position(position)
        if len(self.checking_particles) != 1:
            self.checking_particles = [new_particle]
            return "Pick a second point"

        self.checking_particles.append(new_particle)

        if self.checking_particles[0] == self.checking_particles[1]:
            return "Please choose two separate points"

        return self.checking_particles[0].calculate_params(self.checking_particles[1])

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
            print("Skipping tomo {}, contains no lattices".format(self.name))
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
        self.generate_lattice_dfs()

    @staticmethod
    def scatter3d_trace(
        df: pd.DataFrame, colour="#000000", opacity=1.0
    ) -> go.Scatter3d:
        """
        Produce 3d scatter plot of given dataframe of particles
        """
        return go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            text=df["n"],
            marker=dict(size=6, color=colour, opacity=opacity),
            showlegend=False,
        )

    @staticmethod
    def cone_trace(
        df: pd.DataFrame, colour="#000000", opacity=1, cone_size=10.0
    ) -> go.Cone:
        """
        Produce cone plot of given dataframe of particles
        """
        return go.Cone(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            u=df["u"],
            v=df["v"],
            w=df["w"],
            text=df["n"],
            sizemode="absolute",
            sizeref=cone_size,
            colorscale=[[0, colour], [1, colour]],
            showscale=False,
            opacity=opacity,
        )

    def particles_trace(
        self, df: pd.DataFrame, cone_size=False, cone_fix=True, **kwargs
    ) -> "go.Cone | go.Scatter3d":
        """
        Produce a plotly trace of a dataframe of particles.
        Parameters
        ----------
        df: dataframe of particles
        cone_size: size of cones to plot. A negative value instead plots points.
        cone_fix: force the size of cones to be constant by adding an extra plot.

        Returns
        -------
        Cone or Scatter3d trace of particles
        """
        if cone_size > 0:
            if cone_fix:
                df = pd.concat([df, self.cone_fix])
            return Tomogram.cone_trace(df, cone_size=cone_size, **kwargs)
        else:
            return Tomogram.scatter3d_trace(df, **kwargs)

    def plot_all_particles(self, **kwargs) -> "go.Cone | go.Scatter3d":
        return self.particles_trace(self.all_particles_df(), **kwargs)

    def lattice_trace(self, lattice_id: int, **kwargs) -> "go.Cone | go.Scatter3d":
        return self.particles_trace(self.lattice_df_dict[lattice_id], **kwargs)

    def checking_particle_trace(self, **kwargs):
        return self.particles_trace(
            self.checking_particles_df(), colour=BLACK, opacity=1.0, **kwargs
        )

    def all_lattices_trace(self, showing_removed_particles=False, **kwargs):
        traces = []
        colour_dict = dict()
        hex_vals = colour_range(len(self.lattices))
        tomo_is_uncleaned = len(self.lattices) == 1
        for idx, lattice_key in enumerate(self.lattices.keys()):
            hex_val = WHITE if lattice_key in self.selected_lattices else hex_vals[idx]
            colour_dict.update({lattice_key: hex_val})

        # assign colours and plot lattice
        for lattice_key in self.lattices.keys():
            clean_lattice = lattice_key > 0
            selected_lattice = lattice_key in self.selected_lattices
            if selected_lattice:
                colour = WHITE
                opacity = 1
            elif clean_lattice:
                colour = colour_dict[lattice_key]
                opacity = 1
            else:
                # uncleaned particles
                opacity = 0.6
                if tomo_is_uncleaned:
                    colour = WHITE
                elif showing_removed_particles:
                    colour = BLACK
                else:
                    continue
            traces.append(
                self.lattice_trace(
                    lattice_key, colour=colour, opacity=opacity, **kwargs
                )
            )
        # Checking Particles
        traces.append(self.checking_particle_trace(**kwargs))

        return traces

    def plot_all_lattices(self, **kwargs) -> go.Figure:
        """
        Returns a figure of all lattices in a tomogram

        kwargs:
            "showing_removed_particles": True to show unclean particles. Defaults to False
            "cone_size": size of cones to plot. Set to negative value to instead plot points.
            "cone_fix": True to include a small set of cones which forces all cones to have the same
                size. Defaults to True
        """
        fig = simple_figure()
        traces = self.all_lattices_trace(**kwargs)
        for trace in traces:
            fig.add_trace(trace)
        return fig


def simple_figure() -> go.Figure():
    """
    Returns a simple empty figure with generally appropriate settings for particle display
    """
    layout = go.Layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig = go.Figure(layout=layout)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(scene_aspectmode="data")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10})
    fig["layout"]["uirevision"] = "a"
    return fig


def colour_range(num_points: int) -> list[str]:
    """
    Create an even range of colours across the spectrum

    Parameters
    ----------
    num_points
        Number of colours to create

    Returns
    -------
        List of colours in the form "rgb({},{},{})"
    """
    hsv_tuples = [(x * 1.0 / num_points, 0.75, 0.75) for x in range(num_points)]
    rgb_tuples = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    return [
        "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
        for (r, g, b) in rgb_tuples
    ]


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
