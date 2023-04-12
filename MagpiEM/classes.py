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
from time import time as tm

ADJ_RANGE = (-1, 0, 1)
ADJ_AREA_GEN = tuple([(i, j, k) for i in ADJ_RANGE for j in ADJ_RANGE for k in ADJ_RANGE])


class Cleaner:
    cc_threshold: float
    min_neighbours: int
    min_array_size: int
    dist_range: list
    ori_range: list
    pos_range: list

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
        target_pos,
        pos_tol,
        allow_flips,
    ):
        self.cc_threshold = cc_thresh
        self.min_neighbours = min_neigh
        self.min_array_size = min_array
        self.dist_range = Cleaner.dist_range(target_dist, dist_tol)
        self.ori_range = Cleaner.ang_range_dotprod(target_ori, ori_tol)
        self.pos_range = Cleaner.ang_range_dotprod(target_pos, pos_tol)
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
            *self.dist_range, *self.ori_range, *self.pos_range
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

    # def clean_cc(self, particles):
    #     return {particle for particle in particles if particle.cc_score > self.cc_threshold}

    # def filter_neighbours(self, particle, func, val_range):
    #     particle.neighbours = {
    #         neighbour
    #         for neighbour in particle.neighbours
    #         if within(particle.func(neighbour), val_range)
    #     }

    # def clean_by_neigbours(self, particle, func):
    #     self.filter_neighbours(particle)
    #     if particle in

    # def run_cleaning(self, particles):
    #     particles = self.clean_cc(particles)

    #     particles =

    # def clean_orientation(self, particles):
    #     return {
    #         particle
    #         for particle in particles
    #         if within(particle.dot_direction(neighbour), self.ori_range)
    #     }


class Particle:
    particle_id: int
    cc_score: float
    position: np.ndarray
    direction: np.ndarray

    avg_curvature: float

    subtomo: object

    particles: set()

    regions: dict()
    neighbours: set()

    protein_array: int = 0

    def __init__(self, p_id, cc, position, orientation, subtomo):
        self.particle_id = p_id
        self.cc_score = cc
        self.position = position
        self.direction = normalise(orientation)
        self.subtomo = subtomo
        self.neighbours = set()

    def __hash__(self):
        return int.from_bytes(str(self.position).encode("utf-8"), "little")

    def __str__(self):
        return "x:{:.2f}, y:{:.2f}, z:{:.2f}".format(*self.position)

    def output_dict(self):
        return {"cc": self.cc_score, "pos": self.position, "ori": self.direction}

    def displacement_from(self, particle):
        "Displacement vector between two particles"
        return self.position - particle.position

    def distance2(self, particle):
        "Squared distance between particles"
        displ = self.displacement_from(particle)
        return np.vdot(displ, displ)

    def filter_neighbour_orientation(self, orange, flipped_range):
        good_orientation = set()
        for neighbour in self.neighbours:
            ori = self.dot_direction(neighbour)
            if within(ori, orange):
                good_orientation.add(neighbour)
            elif flipped_range and within(ori, flipped_range):
                good_orientation.add(neighbour)
        self.neighbours = good_orientation

    def filter_curvature(self, pos_range):
        good_curvature = {
            neighbour
            for neighbour in self.neighbours
            if within(self.dot_curvature(neighbour), pos_range)
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

    def dot_direction(self, particle):
        "Dot product of two particles' orientations"
        return Particle.dot_product(self.direction, particle.direction)

    def dot_curvature(self, particle):
        "Dot product of particle's orientation with its displacement from second particle"
        return Particle.dot_product(
            particle.direction, normalise(self.displacement_from(particle))
        )

    def choose_protein_array_new(self, array):
        "Recursively assign particle and all neighbours to array"
        self.set_protein_array(array)
        for neighbour in self.neighbours:
            if not neighbour.protein_array:
                neighbour.choose_protein_array_new(array)

    def set_protein_array(self, ar: int):
        if self.protein_array:
            self.subtomo.protein_arrays[self.protein_array].discard(self)
        self.protein_array = ar
        self.subtomo.protein_arrays[ar].add(self)

    def assimilate_protein_arrays(self, assimilate_arrays: set):
        """
        Combine a set of arrays into a single larger array

        Parameters
        ----------
        assimilate_arrays : set
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # choose a random array to assimilate the rest into
        all_arrays = self.subtomo.protein_arrays
        particles = self.subtomo.all_particles
        assimilate_to = assimilate_arrays.pop()
        for particle in particles:
            if particle.protein_array in assimilate_arrays:
                particle.set_protein_array(assimilate_to)
        for array in assimilate_arrays:
            del all_arrays[array]

    def make_neighbours(self, particle2):
        "Define two particles as neighbours"
        self.neighbours.add(particle2)
        particle2.neighbours.add(self)

    def calculate_params(self, particle2):
        "Return set of useful parameters about two particles"
        distance = self.distance2(particle2) ** 0.5
        orientation = np.degrees(np.arccos(self.dot_direction(particle2)))
        curvature = np.degrees(np.arccos(self.dot_curvature(particle2)))
        return "Distance: {:.1f}\nOrientation: {:.1f}°\nDisplacement: {:.1f}°".format(
            distance, orientation, curvature
        )
    
    @staticmethod
    def from_array(plist, subtomo):
        """
        Produce an array of particles from parameters

        Parameters
        ----------
        plist : List of lists of parameters
            List of particles. Each entry should be a list of parameters
            in the following order:
                cc value, [x, y, z], [u, v, w]
        subtomo: SubTomogram
            SubTomogram object from which the particles are from
        
        Returns
        -------
        Set of particles

        """
        {Particle(idx, *pdata, subtomo) for idx, pdata in enumerate(plist)}
        #def __init__(self, p_id, cc, position, orientation, particle_set, subtomo):
    #__init__(self, p_id, cc, position, orientation, particle_set, subtomo):

    @staticmethod
    def from_geom_mat(subtomo, garr: np.ndarray, cc_thresh):
        "Read set of particles from matlab array"
        particles = set()
        for idx, pdata in enumerate(garr):
            # if pdata[0] < cc_thresh:
            #     print("bad cc", pdata[0])
            #     continue
            new_particle = Particle(
                idx, pdata[0], pdata[10:13], pdata[22:25], particles, subtomo
            )
            particles.add(new_particle)
        return particles

    @staticmethod
    def from_imod(imod_data, subtomo):
        "Read set of particles from imod model"
        particles = set()
        blank_orientation = [0, 0, 1]
        for x in imod_data:
            particles.add(Particle(0, 9999, x, blank_orientation, particles, subtomo))
        return particles

    def find_avg_curvature(self):
        if len(self.neighbours) == 0:
            self.avg_curvature = 0.0
            return
        # print(np.mean([self.dot_curvature(neighbour) for neighbour in self.neighbours]))
        self.avg_curvature = np.mean(
            [self.dot_curvature(neighbour) for neighbour in self.neighbours]
        )


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


class SubTomogram:
    name: str
    all_particles: set()
    current_particles: set()
    auto_cleaned_particles: set()
    removed_particles: set()
    selected_n: set()
    # manual_cleaned_particles: set()

    checking_particles: list

    position_only: bool

    protein_arrays: defaultdict

    particles_fate: defaultdict

    particle_df_dict: dict()
    cone_fix: pd.DataFrame

    # keep_selected: bool

    cleaning_params: Cleaner

    # viewed: bool
    reference_points: set()
    reference_df: set()

    # sample_particle: Particle

    def __init__(self, name):
        self.protein_arrays = defaultdict(lambda: set())
        self.protein_arrays[0] = set()
        self.name = name
        # self.sample_particle = next(iter(particles))
        self.selected_n = set()
        self.auto_cleaned_particles = set()
        self.particles_fate = defaultdict(lambda: set())
        self.reference_points = set()
        self.checking_particles = []
        self.cone_fix = None

    @staticmethod
    def tomo_from_imod(name, imod_struct):
        subtomo = SubTomogram(name)
        subtomo.set_particles(Particle.from_imod(imod_struct, subtomo))
        subtomo.position_only = True
        return subtomo

    # @staticmethod
    # def tomo_from_em(name, em_df):
    #     subtomo = SubTomogram(name)
    #     particles = Particle.from_em(em_df, subtomo)
    #     subtomo.position_only = False
    #     return subtomo

    # @staticmethod
    # def tomo_from_imod(name, imod_filename):
    #     subtomo = SubTomogram(name)
    #     positions = read_write.positions_from_imod(imod_filename)
    #     particles = set()
    #     no_ori = np.array([0.0, 0.0, 1.0])
    #     for x in np.nditer(positions, flags=['external_loop']):
    #         particles.add(Particle(0, 9999, x, no_ori, particles, subtomo))
    #     subtomo.set_particles(particles)
    #     subtomo.position_only = True
    #     return subtomo

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
            "_".join([str(q) for q in np.array(coords) + np.array(adj_direction)])
            for adj_direction in ADJ_AREA_GEN
        ]

    @staticmethod
    def find_nearby_particles(regions, region_key):
        return set().union(
            *[
                regions[k]
                for k in SubTomogram.find_nearby_keys(region_key)
                if k in regions
            ]
        )

    def proximity_clean(self, drange):
        self.auto_cleaned_particles = set()
        prox_range = drange
        # print(prox_range)
        ref_regions = SubTomogram.assign_regions(
            self.reference_points, max(prox_range) ** 0.5
        )
        particle_regions = SubTomogram.assign_regions(
            self.all_particles, max(prox_range) ** 0.5
        )

        for rkey, region in particle_regions.items():
            if len(region) == 0:
                continue
            proximal_refs = SubTomogram.find_nearby_particles(ref_regions, rkey)

            for particle in region:
                for ref in proximal_refs:
                    if within(particle.distance2(ref), prox_range):
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
        keys = set(self.protein_arrays.keys()).copy()
        for n in keys:
            self.delete_array(n)
        self.protein_arrays[0] = set()

    def find_particle_neighbours(self, drange):
        particles = self.all_particles
        t0 = tm()
        regions = SubTomogram.assign_regions(particles, max(drange) ** 0.5)
        print("Assigning particles to regions", tm() - t0)
        # max_dist = max(drange) ** 0.5

        # for particle in particles:
        #     position_list = [str(math.floor(q / max_dist)) for q in particle.position]
        #     locality_id = "_".join(position_list)
        #     regions[locality_id].add(particle)

        t0 = tm()
        for rkey, region in regions.items():
            if len(region) == 0:
                continue
            proximal_particles = SubTomogram.find_nearby_particles(regions, rkey)

            for particle in regions[rkey]:
                for particle2 in proximal_particles:
                    if particle == particle2:
                        continue
                    if within(particle.distance2(particle2), drange):
                        particle.make_neighbours(particle2)

            # remove all checked particles from further checks
            regions[rkey] = set()
        print("Finding particles in regions", tm() - t0)

    def __hash__(self):
        return self.name

    def __str__(self):
        return "Subtomogram {}, containing {} particles, {} after auto cleaning. Selected arrays: {}".format(
            self.name,
            len(self.all_particles),
            len(self.auto_cleaned_particles),
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

    def selected_particle_ids(self):
        return {
            particle.particle_id
            for particle in self.auto_cleaned_particles
            if particle.protein_array in self.selected_n
        }

    def assign_ref_imod(self, imod_data):
        self.reference_points = Particle.from_imod(imod_data, self)
        particle_data = [
            [*particle.position, *particle.direction, particle.protein_array]
            for particle in self.reference_points
        ]
        self.reference_df = pd.DataFrame(
            particle_data, columns=["x", "y", "z", "u", "v", "w", "n"]
        )

    def unselected_particle_ids(self):
        return {
            particle.particle_id
            for particle in self.auto_cleaned_particles
            if not particle.protein_array in self.selected_n
        }

    @staticmethod
    def particles_to_df(particles):
        particle_data = [
            [*particle.position, *particle.direction, particle.protein_array]
            for particle in particles
        ]
        return pd.DataFrame(particle_data, columns=["x", "y", "z", "u", "v", "w", "n"])

    def all_particles_df(self):
        return SubTomogram.particles_to_df(self.all_particles)

    def nonchecking_particles_df(self):
        unchecking = self.all_particles.difference(set(self.checking_particles))
        return SubTomogram.particles_to_df(unchecking)

    def checking_particles_df(self):
        return SubTomogram.particles_to_df(set(self.checking_particles))

    def autoclean(self):
        self.all_particles = {
            particle
            for particle in self.all_particles
            if particle.cc_score > self.cleaning_params.cc_threshold
        }
        self.find_particle_neighbours(self.cleaning_params.dist_range)
        t0 = tm()
        for particle in self.all_particles:
            neighbours = particle.neighbours

            if len(neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["low_neighbours"].add(particle)
                continue
        print("Finding low neighbours", tm() - t0)
        t0 = tm()
        for particle in self.all_particles:  # temp for timing
            # check correct orientation
            particle.filter_neighbour_orientation(
                self.cleaning_params.ori_range, self.cleaning_params.flipped_ori_range
            )
            if len(particle.neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["wrong_ori"].add(particle)
                continue
        print("Comparing angles", tm() - t0)
        t0 = tm()
        for particle in self.all_particles:  # temp for timing
            # check correct positioning
            particle.filter_curvature(self.cleaning_params.pos_range)
            if len(particle.neighbours) < self.cleaning_params.min_neighbours:
                self.particles_fate["wrong_disp"].add(particle)
                continue
        print("Comparing curvatures", tm() - t0)
        t0 = tm()
        for particle in self.all_particles:  # temp for timing
            if particle.protein_array:
                continue
            # if good, assign to protein array
            # particle.choose_protein_array()
            particle.choose_protein_array_new(len(self.protein_arrays))

        print("Choosing initial arrays", tm() - t0)
        # can't check size of array until all particles allocated

        # can't delete arrays within loop, changes size of dict
        t0 = tm()
        bad_arrays = set()
        for akey, protein_array in self.protein_arrays.items():
            if len(protein_array) < self.cleaning_params.min_array_size:
                bad_arrays.add(akey)
                for particle in protein_array:
                    self.particles_fate["small_array"].add(particle)
            else:
                for particle in protein_array:
                    particle.find_avg_curvature()
                    self.auto_cleaned_particles.add(particle)
        for akey in bad_arrays:
            self.delete_array(akey)
        print("Time to assimilate arrays", tm() - t0)

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
        else:
            self.selected_n.add(n)

    def get_convex_arrays(self):
        return {
            a_id
            for a_id, particles in self.protein_arrays.items()
            if a_id > 0
            and np.mean([particle.avg_curvature for particle in particles]) < 0
        }

    def toggle_convex_arrays(self):
        for a_id in self.get_convex_arrays():
            self.toggle_selected(a_id)

    def toggle_concave_arrays(self):
        for a_id in set(self.protein_arrays.keys()).difference(
            self.get_convex_arrays()
        ):
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

    def delete_array(self, n):
        # can't change size of array while iterating
        array_particles = self.protein_arrays[n].copy()
        for particle in array_particles:
            particle.set_protein_array(0)
            self.auto_cleaned_particles.discard(particle)
        self.protein_arrays.pop(n)

    def write_prog_dict(self):
        if len(self.protein_arrays.keys()) < 2:
            print("Skipping tomo {}, contains no arrays").format(self.name)
            return
        arr_dict = {}
        for ind, arr in self.protein_arrays.items():
            # must use lists, or yaml interprets as dicts
            if ind == 0:
                continue
            # print("ind", ind)
            p_ids = [p.particle_id for p in arr]
            arr_dict[ind] = p_ids
        arr_dict["selected"] = list(self.selected_n)
        return arr_dict

    @staticmethod
    def from_prog_dict(name, mat_geomt, prog_dict):
        print("loading tomo", name)
        subtomo = SubTomogram.tomo_from_mat(name, mat_geomt)
        inverted_prog_dict = {}
        for array, particles in prog_dict.items():
            if array == "selected":
                continue
            for particle in particles:
                inverted_prog_dict[particle] = array
        for particle in subtomo.all_particles:
            p_id = particle.particle_id
            if p_id in inverted_prog_dict.keys():
                particle.set_protein_array(
                    int(inverted_prog_dict[particle.particle_id])
                )
            else:
                particle.set_protein_array(0)
        subtomo.selected_n = set(prog_dict["selected"])
        subtomo.generate_particle_df()
        return subtomo


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
