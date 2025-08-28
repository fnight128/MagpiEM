# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""

import math
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Union

from .cleaner import Cleaner
from .particle import Particle
from .utilities import within
from .plotting_helpers import (
    simple_figure,
    colour_range,
    create_scatter_trace,
    create_cone_traces,
    generate_cone_fix_points,
    append_cone_fix_to_lattice,
)

WHITE = "#FFFFFF"
GREY = "#646464"
BLACK = "#000000"


class Tomogram:
    name: str
    all_particles: set
    removed_particles: set
    selected_lattices: set

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

    def autoclean(self) -> dict:
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
                for particle in lattice:
                    self.particles_fate["small_array"].add(particle)
        for lattice_key in bad_lattices:
            self.delete_lattice(lattice_key)
        return self.write_progress_dict()

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
        self.generate_lattice_dfs()

    @staticmethod
    def scatter3d_trace(
        df: pd.DataFrame, colour="#000000", opacity=1.0, **kwargs
    ) -> "go.Scatter3d":
        """
        Produce 3d scatter plot of given dataframe of particles
        """
        positions = df[["x", "y", "z"]].values
        lattice_id = df["n"].iloc[0] if len(df) > 0 else 0
        return create_scatter_trace(positions, colour, opacity, lattice_id, **kwargs)

    @staticmethod
    def cone_trace(
        df: pd.DataFrame, colour="#000000", opacity=1, cone_size=10.0, **kwargs
    ) -> "go.Cone":
        """
        Produce cone plot of given dataframe of particles
        """
        positions = df[["x", "y", "z"]].values
        orientations = df[["u", "v", "w"]].values
        lattice_id = df["n"].iloc[0] if len(df) > 0 else 0
        return create_cone_traces(
            positions, orientations, cone_size, colour, opacity, lattice_id, **kwargs
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
            positions = df[["x", "y", "z"]].values
            orientations = df[["u", "v", "w"]].values
            lattice_id = df["n"].iloc[0] if len(df) > 0 else 0

            if cone_fix:
                # Generate cone fix points for this tomogram
                cone_fix_positions, cone_fix_orientations = generate_cone_fix_points(
                    [
                        [pos.tolist(), ori.tolist()]
                        for pos, ori in zip(positions, orientations)
                    ]
                )
                positions, orientations = append_cone_fix_to_lattice(
                    positions, orientations, cone_fix_positions, cone_fix_orientations
                )

            return create_cone_traces(positions, orientations, cone_size, **kwargs)
        else:
            return Tomogram.scatter3d_trace(df, **kwargs)

    def plot_all_particles(self, **kwargs) -> "go.Cone | go.Scatter3d":
        return self.particles_trace(self.all_particles_df(), **kwargs)

    def lattice_trace(self, lattice_id: int, **kwargs) -> "go.Cone | go.Scatter3d":
        return self.particles_trace(
            self.lattice_df_dict[lattice_id], name=lattice_id, **kwargs
        )

    def all_lattices_trace(self, showing_removed_particles=False, **kwargs):
        traces = []
        colour_dict = dict()
        hex_vals = colour_range(len(self.lattices))
        tomo_is_uncleaned = len(self.lattices) == 1

        # Generate cone fix points for this tomogram
        all_particles_data = [
            [particle.position.tolist(), particle.orientation.tolist()]
            for particle in self.all_particles
        ]
        cone_fix_positions, cone_fix_orientations = generate_cone_fix_points(
            all_particles_data
        )

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

            # Get particles for this lattice
            lattice_particles = self.lattices[lattice_key]
            if len(lattice_particles) == 0:
                continue

            # Extract positions and orientations
            positions = np.array([particle.position for particle in lattice_particles])
            orientations = np.array(
                [particle.orientation for particle in lattice_particles]
            )

            cone_size = kwargs.get("cone_size", 0)
            if cone_size > 0:
                # Add cone fix points
                positions, orientations = append_cone_fix_to_lattice(
                    positions, orientations, cone_fix_positions, cone_fix_orientations
                )
                trace = create_cone_traces(
                    positions, orientations, cone_size, colour, opacity, lattice_key
                )
            else:
                trace = create_scatter_trace(positions, colour, opacity, lattice_key)

            trace.name = f"Lattice {lattice_key}"
            traces.append(trace)

        return traces

    def plot_all_lattices(self, **kwargs) -> "go.Figure":
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
