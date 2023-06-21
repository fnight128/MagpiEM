# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:54:48 2022

@author: Frank
"""
import numpy as np
import pandas as pd
import math
from prettytable import PrettyTable
from collections import defaultdict

ADJ_RANGE = (-1, 0, 1)
ADJ_AREA_GEN = tuple(
    [(i, j, k) for i in ADJ_RANGE for j in ADJ_RANGE for k in ADJ_RANGE]
)


class Cleaner:
    cc_threshold: float
    min_neighbours: int
    min_lattice_size: int
    dist_range: list
    ori_range: list
    curv_range: list

    flipped_ori_range: list

    dict_to_print: dict

    def __init__(
        self,
        cc_thresh,
        min_neigh,
        min_array,
        target_dist,
        dist_tol,
        target_ori,
        ori_tol,
        target_curv,
        curv_tol,
        allow_flips,
    ):
        self.cc_threshold = cc_thresh
        self.min_neighbours = min_neigh
        self.min_lattice_size = min_array
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
            "cc threshold": cc_thresh,
            "min neighbours": min_neigh,
            "min array size": min_array,
            "allow flips": allow_flips,
        }

    def __str__(self):
        return "Allowed distances: {}-{}. Allowed orientations:{}-{}. Allowed curvatures:{}-{}.".format(
            *self.dist_range, *self.ori_range, *self.curv_range
        )

    @staticmethod
    def within(value, allowed_range: tuple):
        return allowed_range[0] <= value <= allowed_range[1]

    @staticmethod
    def dist_range(target_dist, dist_tol):
        dist_tol = abs(dist_tol)
        assert target_dist != 0, "Target distance cannot be 0"
        if target_dist < 0:
            target_dist = abs(target_dist)
            print("Target distance must be > 0, correcting to ", target_dist)
        return [
            (target_dist - dist_tol) ** 2
            if dist_tol < target_dist
            else 0.0001 * dist_tol,
            (target_dist + dist_tol) ** 2,
        ]

    @staticmethod
    def ang_range_dotprod(angle_ideal, angle_tolerance):
        # print("ideal angle: ", angle_ideal, "tolerance: ", angle_tolerance)
        if not within(angle_ideal, [0, 180]):
            angle_ideal = angle_ideal % 180
            print("Angle between adjacent particles must be between 0 and 180 degrees")
            print("Corrected angle: ", angle_ideal)
        elif not within(angle_tolerance, [0, 180]):
            angle_tolerance = angle_tolerance % 180
            print("Angle tolerance must be between 0 and 180 degrees")
            print("Corrected tolerance: ", angle_tolerance)
        min_ang = angle_ideal - angle_tolerance
        max_ang = angle_ideal + angle_tolerance

        # edge cases where tolerance extends beyond [0,180], must fix or range
        # will be unintentionally very small due to cos non-monotonicity
        if min_ang < 0:
            max_ang = max(max_ang, -min_ang)
            min_ang = 0
        elif max_ang > 180:
            min_ang = min(min_ang, 360 - max_ang)
            max_ang = 180

        # print("result:", [np.degrees(np.arccos(np.cos(np.radians(ang)))) for ang in [max_ang, min_ang]])

        return [np.cos(np.radians(ang)) for ang in [max_ang, min_ang]]


class Particle:
    particle_id: int
    cc_score: float
    position: np.ndarray
    orientation: np.ndarray

    tomo: object

    particles: set()

    regions: dict()
    neighbours: set()

    lattice: int = 0

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
        
        Returns
        -------
        Dict of properties about particle:
            "cc": cc score
            "pos": position
            "ori": orientation
        """
        return {"cc": self.cc_score, "pos": self.position, "ori": self.orientation}

    def displacement_from(self, particle):
        """Displacement vector between two particles"""
        return self.position - particle.position

    def distance_sq(self, particle):
        """Squared distance between particles"""
        disp = self.displacement_from(particle)
        return np.vdot(disp, disp)

    def filter_neighbour_orientation(self, orange, flipped_range):
        good_orientation = set()
        for neighbour in self.neighbours:
            ori = self.dot_orientation(neighbour)
            if within(ori, orange):
                good_orientation.add(neighbour)
            elif flipped_range and within(ori, flipped_range):
                good_orientation.add(neighbour)
        self.neighbours = good_orientation

    def filter_curvature(self, curv_range):
        good_curvature = {
            neighbour
            for neighbour in self.neighbours
            if within(self.dot_curvature(neighbour), curv_range)
        }
        self.neighbours = good_curvature

    @staticmethod
    def dot_product(v1, v2):
        "Dot product of two vectors. Fixed to 1 for anomalous high values"
        # temp fix for values appearing to be > 1
        dot = np.vdot(v1, v2)
        if dot > 1:
            dot = 1
        return dot

    def dot_orientation(self, particle):
        """Dot product of two particles' orientations"""
        return Particle.dot_product(self.orientation, particle.orientation)

    def dot_curvature(self, particle):
        """Dot product of particle's orientation with its displacement from second particle"""
        return Particle.dot_product(
            particle.orientation, normalise(self.displacement_from(particle))
        )

    def choose_new_lattice(self, lattice):
        """Recursively assign particle and all neighbours to lattice"""
        self.set_lattice(lattice)
        for neighbour in self.neighbours:
            if not neighbour.lattice:
                neighbour.choose_new_lattice(lattice)

    def set_lattice(self, ar: int):
        if self.lattice:
            self.tomo.lattices[self.lattice].discard(self)
        self.lattice = ar
        self.tomo.lattices[ar].add(self)

    def assimilate_lattices(self, assimilate_lattices: set):
        """
        Combine a set of lattices into a single larger lattice

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

    def make_neighbours(self, particle2):
        """Define two particles as neighbours"""
        self.neighbours.add(particle2)
        particle2.neighbours.add(self)

    def calculate_params(self, particle2):
        """Return set of useful parameters about two particles"""
        distance = self.distance_sq(particle2) ** 0.5
        orientation = np.degrees(np.arccos(self.dot_orientation(particle2)))
        curvature = np.degrees(np.arccos(self.dot_curvature(particle2)))
        return "Distance: {:.1f}\nOrientation: {:.1f}°\nDisplacement: {:.1f}°".format(
            distance, orientation, curvature
        )

    @staticmethod
    def from_array(plist, tomo, ids=None):
        """
        Produce a set of particles from parameters

        Parameters
        ----------
        plist : List of lists of parameters
            List of particles. Each entry in the list
            should be a list of parameters in the
            following order:
                cc value, [x, y, z], [u, v, w]
        tomo: tomogram
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

    def get_avg_curvature(self):
        if len(self.neighbours) == 0:
            return 0.0
        return np.mean([self.dot_curvature(neighbour) for neighbour in self.neighbours])


class ReferenceParticle(Particle):
    def __eq__(self, other):
        """
        Define two particles as equal if in same position
        Rounded as very small differences often introduced
        by e.g. binning
        """
        self_rough_pos = np.round(self.position, 2)
        other_rough_pos = np.round(other.position, 2)
        return (self_rough_pos == other_rough_pos).all()


class tomogram:
    name: str
    all_particles: set()
    current_particles: set()
    auto_cleaned_particles: set()
    removed_particles: set()
    selected_n: set()

    checking_particles: list

    position_only: bool

    lattices: defaultdict

    particles_fate: defaultdict

    particle_df_dict: dict()
    cone_fix: pd.DataFrame

    cleaning_params: Cleaner

    reference_points: set()
    reference_df: set()

    def __init__(self, name):
        self.lattices = defaultdict(lambda: set())
        self.lattices[0] = set()
        self.name = name
        self.selected_n = set()
        self.auto_cleaned_particles = set()
        self.particles_fate = defaultdict(lambda: set())
        self.reference_points = set()
        self.checking_particles = []
        self.cone_fix = None

    @staticmethod
    def assign_regions(particles: set, max_dist: float):
        regions = defaultdict(lambda: set())
        for particle in particles:
            position_list = [str(math.floor(q / max_dist)) for q in particle.position]
            locality_id = "_".join(position_list)
            regions[locality_id].add(particle)
        return regions

    @staticmethod
    def find_nearby_keys(region_key):
        coords = [int(q) for q in region_key.split("_")]
        return [
            "_".join([str(q) for q in np.array(coords) + np.array(adj_orientation)])
            for adj_orientation in ADJ_AREA_GEN
        ]

    @staticmethod
    def find_nearby_particles(regions, region_key):
        return set().union(
            *[regions[k] for k in tomogram.find_nearby_keys(region_key) if k in regions]
        )

    def proximity_clean(self, drange):
        self.auto_cleaned_particles = set()
        prox_range = drange
        # print(prox_range)
        ref_regions = tomogram.assign_regions(
            self.reference_points, max(prox_range) ** 0.5
        )
        particle_regions = tomogram.assign_regions(
            self.all_particles, max(prox_range) ** 0.5
        )

        for rkey, region in particle_regions.items():
            if len(region) == 0:
                continue
            proximal_refs = tomogram.find_nearby_particles(ref_regions, rkey)

            for particle in region:
                for ref in proximal_refs:
                    if within(particle.distance_sq(ref), prox_range):
                        self.auto_cleaned_particles.add(particle)
                        break
        print("cleaned particles: ", len(self.auto_cleaned_particles))

        particle_data = [
            [*particle.position, 1] for particle in self.auto_cleaned_particles
        ]
        self.particle_df_dict = {
            1: pd.DataFrame(particle_data, columns=["x", "y", "z", "n"])
        }

    def reset_cleaning(self):
        keys = set(self.lattices.keys()).copy()
        for n in keys:
            self.delete_lattice(n)
        self.lattices[0] = set()

    def find_particle_neighbours(self, drange):
        particles = self.all_particles
        # t0 = tm()
        regions = tomogram.assign_regions(particles, max(drange) ** 0.5)
        # print("Assigning particles to regions", tm() - t0)

        # t0 = tm()
        for rkey, region in regions.items():
            if len(region) == 0:
                continue
            proximal_particles = tomogram.find_nearby_particles(regions, rkey)

            for particle in regions[rkey]:
                for particle2 in proximal_particles:
                    if particle == particle2:
                        continue
                    if within(particle.distance_sq(particle2), drange):
                        particle.make_neighbours(particle2)

            # remove all checked particles from further checks
            regions[rkey] = set()
        # print("Finding particles in regions", tm() - t0)

    def __hash__(self):
        return self.name

    def __str__(self):
        return "tomogram {}, containing {} particles. Selected lattices: {}".format(
            self.name,
            len(self.all_particles),
            ",".join({str(n) for n in self.selected_n}),
        )

    def get_particle_from_position(self, position):
        rough_pos = np.around(position, decimals=0)
        for particle in self.all_particles:
            if all(np.around(particle.position, decimals=0) == rough_pos):
                return particle
        return 0

    def set_particles(self, particles):
        self.all_particles = particles

    def set_clean_params(self, cp):
        self.cleaning_params = cp

    def selected_particle_ids(self, selected=True):
        return {
            particle.particle_id
            for particle in self.auto_cleaned_particles
            if (particle.lattice in self.selected_n) == selected
        }

    def assign_ref_imod(self, imod_data):
        self.reference_points = Particle.from_imod(imod_data, self)
        particle_data = [
            [*particle.position, *particle.orientation, particle.lattice]
            for particle in self.reference_points
        ]
        self.reference_df = pd.DataFrame(
            particle_data, columns=["x", "y", "z", "u", "v", "w", "n"]
        )

    @staticmethod
    def particles_to_df(particles):
        particle_data = [
            [*particle.position, *particle.orientation, particle.lattice]
            for particle in particles
        ]
        return pd.DataFrame(particle_data, columns=["x", "y", "z", "u", "v", "w", "n"])

    def all_particles_df(self):
        return tomogram.particles_to_df(self.all_particles)

    def nonchecking_particles_df(self):
        unchecking = self.all_particles.difference(set(self.checking_particles))
        return tomogram.particles_to_df(unchecking)

    def checking_particles_df(self):
        return tomogram.particles_to_df(set(self.checking_particles))

    def autoclean(self):
        self.all_particles = {
            particle
            for particle in self.all_particles
            if particle.cc_score > self.cleaning_params.cc_threshold
        }
        self.find_particle_neighbours(self.cleaning_params.dist_range)
        # t0 = tm()
        for particle in self.all_particles:
            neighbours = particle.neighbours

            if len(neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["low_neighbours"].add(particle)
                continue
        # print("Finding low neighbours", tm() - t0)
        # t0 = tm()
        for particle in self.all_particles:  # temp for timing
            # check correct orientation
            particle.filter_neighbour_orientation(
                self.cleaning_params.ori_range, self.cleaning_params.flipped_ori_range
            )
            if len(particle.neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["wrong_ori"].add(particle)
                continue
        # print("Comparing angles", tm() - t0)
        # t0 = tm()
        for particle in self.all_particles:  # temp for timing
            # check correct positioning
            particle.filter_curvature(self.cleaning_params.curv_range)
            if len(particle.neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["wrong_disp"].add(particle)
                continue
        # print("Comparing curvatures", tm() - t0)
        # t0 = tm()
        for particle in self.all_particles:  # temp for timing
            if particle.lattice:
                continue
            # if good, assign to protein array
            # particle.choose_lattice()
            particle.choose_new_lattice(len(self.lattices))

        bad_lattices = set()
        for lkey, lattice in self.lattices.items():
            if len(lattice) < self.cleaning_params.min_lattice_size:
                bad_lattices.add(lkey)
                for particle in lattice:
                    self.particles_fate["small_array"].add(particle)
            else:
                for particle in lattice:
                    self.auto_cleaned_particles.add(particle)
        for lkey in bad_lattices:
            self.delete_lattice(lkey)

    def particle_fate_table(self):
        pf = self.particles_fate
        out_table = PrettyTable()
        out_table.field_names = ["Category", "Particles"]
        out_table.add_rows(
            [
                ["________Total_________", len(self.all_particles)],
                # ["Low CC Score", below_CC_count],
                ["Low Neighbours", len(pf["low_neighbours"])],
                ["Wrong Neighbour Orientation", len(pf["wrong_ori"])],
                ["Wrong Neighbour Position", len(pf["wrong_disp"])],
                ["Small Array", len(pf["small_array"])],
                ["Good Particles", len(self.auto_cleaned_particles)],
                # ["Total Removed", len(self.all_particles) - len(self.manual_cleaned_particles) if(self.manual_cleaned_particles) else "unfinished"]
            ]
        )
        return out_table

    def cone_fix_df(self):
        if self.cone_fix is not None:
            return self.cone_fix

        particle_df = self.all_particles_df()
        # see cone_fix_readme.txt for explanation

        # locate lower corner of plot, and find overall size
        max_series = particle_df.max()
        min_series = particle_df.min()

        range_series = max_series.subtract(min_series)
        range_magnitude = math.hypot(
            range_series["x"], range_series["y"], range_series["z"]
        )

        # create two extremely small vectors extremely close together
        scaling = 10000
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

    def generate_particle_df(self):
        particle_df = self.all_particles_df()
        # split into one dataframe for each array
        self.particle_df_dict = dict(iter(particle_df.groupby("n")))

    def toggle_selected(self, n):
        if n in self.selected_n:
            self.selected_n.remove(n)
        # ensure 0 is never selected - represents unclean particles
        elif n != 0:
            self.selected_n.add(n)

    def get_convex_arrays(self):
        return {
            a_id
            for a_id, particles in self.lattices.items()
            if np.mean([particle.get_avg_curvature() for particle in particles]) < 0
        }

    def get_concave_arrays(self):
        return set(self.lattices.keys()).difference(self.get_convex_arrays())

    def toggle_convex_arrays(self):
        for a_id in self.get_convex_arrays():
            self.toggle_selected(a_id)

    def toggle_concave_arrays(self):
        for a_id in self.get_concave_arrays():
            self.toggle_selected(a_id)

    def show_particle_data(self, position):
        new_particle = self.get_particle_from_position(position)
        if len(self.checking_particles) != 1:
            self.checking_particles = [new_particle]
            return "Pick a second point"

        self.checking_particles.append(new_particle)

        if self.checking_particles[0] == self.checking_particles[1]:
            return "Please choose two separate points"

        return self.checking_particles[0].calculate_params(self.checking_particles[1])

    def delete_lattice(self, n):
        # can't change size of array while iterating
        array_particles = self.lattices[n].copy()
        for particle in array_particles:
            particle.set_lattice(0)
            self.auto_cleaned_particles.discard(particle)
        self.lattices.pop(n)

    def write_prog_dict(self):
        if len(self.lattices.keys()) < 2:
            print("Skipping tomo {}, contains no arrays").format(self.name)
            return
        arr_dict = {}
        for ind, arr in self.lattices.items():
            # must use lists, or yaml interprets as dicts
            if ind == 0:
                continue

            p_ids = [p.particle_id for p in arr]
            arr_dict[ind] = p_ids
        selected_lattices = list(self.selected_n)
        arr_dict["selected"] = selected_lattices
        return arr_dict

    def apply_prog_dict(self, prog_dict):
        # invert dict
        inverted_prog_dict = {}
        for array, particles in prog_dict.items():
            # skip dict which stores which arrays are selected
            if array == "selected":
                continue
            for particle in particles:
                inverted_prog_dict[particle] = array
        # using inverted dict, assign all particles to lattices
        for particle in self.all_particles:
            p_id = particle.particle_id
            if p_id in inverted_prog_dict.keys():
                particle.set_lattice(
                    int(inverted_prog_dict[particle.particle_id])
                )
                self.auto_cleaned_particles.add(particle)
            else:
                particle.set_lattice(0)
        self.selected_n = set(prog_dict["selected"])
        self.generate_particle_df()


def normalise(vec: np.ndarray):
    """Normalise vector of any dimension"""
    assert not all(x == 0 for x in vec), "Attempted to normalise a zero vector"
    mag = np.linalg.norm(vec)
    return vec / mag


def within(value: float, allowed_range: tuple):
    """


    Parameters
    ----------
    value : float
        Value to check
    allowed_range : tuple
        Tuple consisting of (min_val, max_val)
        Tuple MUST be ordered, this is not validated

    Returns
    -------
    bool
        allowed_range[0] <= value <= allowed_range[1]

    """
    return allowed_range[0] <= value <= allowed_range[1]
