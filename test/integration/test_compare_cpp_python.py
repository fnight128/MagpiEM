import ctypes
import pathlib
import numpy as np
import time
import sys
import logging
import pytest

# Add the parent directory to the path to ensure we use the local magpiem module
current_dir = pathlib.Path(__file__).parent.absolute()
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

from magpiem.io.io_utils import read_emc_mat, read_single_tomogram  # noqa: E402
from magpiem.processing.classes.tomogram import Tomogram  # noqa: E402
from magpiem.processing.classes.cleaner import Cleaner  # noqa: E402
from magpiem.processing.cpp_integration import (  # noqa: E402
    setup_cpp_library,
    clean_tomo_with_cpp,
)  # noqa: E402
from ..test_utils import (  # noqa: E402
    TestConfig,
    get_test_data_path,
    ensure_test_data_generated,
)

logger = logging.getLogger(__name__)

ensure_test_data_generated()


@pytest.fixture
def c_lib():
    """Fixture to provide the C++ library."""
    return setup_cpp_library()


@pytest.fixture
def test_data():
    """Fixture to provide test particle data."""
    return load_particle_data()


@pytest.fixture
def test_cleaner():
    """Fixture to provide test cleaner object."""
    return Cleaner.from_user_params(*TEST_CLEANER_VALUES)


@pytest.fixture
def python_reference(test_data, test_cleaner):
    """Fixture to provide Python reference results."""
    # Calculate Python reference
    reference_dict, _ = calculate_python_reference(test_data, test_cleaner)
    return reference_dict


@pytest.fixture
def min_dist(test_cleaner):
    """Fixture to provide minimum distance parameter."""
    return test_cleaner.dist_range[0]


@pytest.fixture
def max_dist(test_cleaner):
    """Fixture to provide maximum distance parameter."""
    return test_cleaner.dist_range[1]


@pytest.fixture
def min_ori(test_cleaner):
    """Fixture to provide minimum orientation parameter."""
    return test_cleaner.ori_range[0]


@pytest.fixture
def max_ori(test_cleaner):
    """Fixture to provide maximum orientation parameter."""
    return test_cleaner.ori_range[1]


@pytest.fixture
def min_curv(test_cleaner):
    """Fixture to provide minimum curvature parameter."""
    return test_cleaner.curv_range[0]


@pytest.fixture
def max_curv(test_cleaner):
    """Fixture to provide maximum curvature parameter."""
    return test_cleaner.curv_range[1]


# Updated paths for test folder location
TEST_DATA_FILE = get_test_data_path(TestConfig.TEST_DATA_STANDARD)
TEST_TOMO_NAME = TestConfig.TEST_TOMO_STANDARD
# appropriate values for test tomogram
TEST_CLEANER_VALUES = TestConfig.TEST_CLEANER_VALUES


def load_particle_data() -> np.ndarray:
    """Load particle data from the test file"""
    test_data = np.array(read_emc_mat(str(TEST_DATA_FILE))[TEST_TOMO_NAME], dtype=float)
    test_data = test_data[
        :, [10, 11, 12, 22, 23, 24]
    ]  # Extract position and orientation columns
    return test_data


def load_test_data() -> (
    tuple[
        np.ndarray, Cleaner, tuple[float, float, float, float, float, float, int, int]
    ]
):
    """Load test data and create Cleaner object"""
    test_cleaner = Cleaner.from_user_params(*TEST_CLEANER_VALUES)

    # Get parameters
    min_dist, max_dist = test_cleaner.dist_range
    min_ori, max_ori = test_cleaner.ori_range
    min_curv, max_curv = test_cleaner.curv_range
    min_lattice_size = test_cleaner.min_lattice_size
    min_neighbours = test_cleaner.min_neighbours

    # Load particle data
    test_data = load_particle_data()

    logger.info(f"Testing with {len(test_data)} particles")

    return (
        test_data,
        test_cleaner,
        (
            min_dist,
            max_dist,
            min_ori,
            max_ori,
            min_curv,
            max_curv,
            min_lattice_size,
            min_neighbours,
        ),
    )


def create_test_cleaner() -> Cleaner:
    return Cleaner.from_user_params(*TEST_CLEANER_VALUES)


def create_test_tomogram() -> Tomogram:
    return read_single_tomogram(str(TEST_DATA_FILE), TEST_TOMO_NAME)


def setup_test_tomogram() -> Tomogram:
    """Create and setup a test tomogram with the given cleaner parameters"""
    test_tomo = create_test_tomogram()
    test_cleaner = create_test_cleaner()
    test_tomo.set_clean_params(test_cleaner)
    test_tomo.find_particle_neighbours(test_cleaner.dist_range)
    return test_tomo


def run_cpp_test(
    c_lib,
    test_data: np.ndarray,
    test_name: str,
    python_reference: list[int],
    test_func,
    *args,
) -> tuple[list[int], float]:
    """Generic function to run C++ tests with common setup and validation"""
    logger.info(f"TEST {test_name}")

    flat_data = [val for particle in test_data for val in particle]
    c_array = (ctypes.c_float * len(flat_data))(*flat_data)
    results_array = (ctypes.c_int * len(test_data))()

    start_time = time.time()
    test_func(c_lib, c_array, len(test_data), results_array, *args)
    test_time = time.time() - start_time

    counts_cpp = [results_array[i] for i in range(len(test_data))]

    # Validate against Python reference
    verify_counts(python_reference, counts_cpp, f"{test_name} filtering")

    return counts_cpp, test_time


def create_test_function(cpp_func_name: str, setup_steps: list[str] = None) -> callable:
    """Factory function to create test functions with common patterns"""

    def test_func(c_lib, c_array, num_particles, results_array, *args):
        # Apply setup steps if provided
        if setup_steps:
            for step in setup_steps:
                if step == "find_neighbours":
                    c_lib.find_neighbours(
                        c_array, num_particles, args[0], args[1], results_array
                    )
                elif step == "filter_by_orientation":
                    c_lib.filter_by_orientation(
                        c_array, num_particles, args[2], args[3], results_array
                    )
                elif step == "filter_by_curvature":
                    c_lib.filter_by_curvature(
                        c_array, num_particles, args[2], args[3], results_array
                    )
        else:
            # Direct function call
            getattr(c_lib, cpp_func_name)(c_array, num_particles, *args, results_array)

    return test_func


def test_cpp_distance_only(
    c_lib,
    test_data: np.ndarray,
    min_dist: float,
    max_dist: float,
    python_reference: list[int],
):
    """Test C++ distance-only neighbour finding"""

    def distance_test(c_lib, c_array, num_particles, results_array, min_dist, max_dist):
        # Convert numpy array to Python list for the C++ extension
        data_list = [float(val) for val in c_array]
        results = c_lib.find_neighbours(data_list, num_particles, min_dist, max_dist)
        # Copy results back to the results array
        for i in range(num_particles):
            results_array[i] = results[i]

    run_cpp_test(
        c_lib,
        test_data,
        "1: Distance-only neighbour finding",
        python_reference["initial"],
        distance_test,
        min_dist,
        max_dist,
    )


def test_cpp_orientation_only(
    c_lib,
    test_data: np.ndarray,
    min_dist: float,
    max_dist: float,
    min_ori: float,
    max_ori: float,
    python_reference: list[int],
):
    """Test C++ orientation filtering alone (after distance filtering)"""

    def orientation_test(
        c_lib,
        c_array,
        num_particles,
        results_array,
        min_dist,
        max_dist,
        min_ori,
        max_ori,
    ):
        # Convert numpy array to Python list for the C++ extension
        data_list = [float(val) for val in c_array]
        # find neighbours
        c_lib.find_neighbours(data_list, num_particles, min_dist, max_dist)
        # filter by orientation
        results = c_lib.filter_by_orientation(
            data_list, num_particles, min_ori, max_ori
        )
        # Copy results back
        for i in range(num_particles):
            results_array[i] = results[i]

    run_cpp_test(
        c_lib,
        test_data,
        "2: Orientation filtering",
        python_reference["orientation"],
        orientation_test,
        min_dist,
        max_dist,
        min_ori,
        max_ori,
    )


def test_cpp_curvature_only(
    c_lib,
    test_data: np.ndarray,
    min_dist: float,
    max_dist: float,
    min_curv: float,
    max_curv: float,
    python_reference: list[int],
):
    """Test C++ curvature filtering alone (after distance filtering)"""

    def curvature_test(
        c_lib,
        c_array,
        num_particles,
        results_array,
        min_dist,
        max_dist,
        min_curv,
        max_curv,
    ):
        # Convert numpy array to Python list for the C++ extension
        data_list = [float(val) for val in c_array]
        # find neighbours
        c_lib.find_neighbours(data_list, num_particles, min_dist, max_dist)
        # filter by curvature
        results = c_lib.filter_by_curvature(
            data_list, num_particles, min_curv, max_curv
        )
        # Copy results back
        for i in range(num_particles):
            results_array[i] = results[i]

    run_cpp_test(
        c_lib,
        test_data,
        "3: Curvature filtering",
        python_reference["curvature"],
        curvature_test,
        min_dist,
        max_dist,
        min_curv,
        max_curv,
    )


def test_cpp_full_pipeline(
    test_data: np.ndarray,
    test_cleaner: Cleaner,
    python_reference: list[int],
):
    """Test C++ full pipeline"""
    logger.info("TEST 4: Full pipeline")

    # Convert test_data to the format expected by cpp_integration
    # test_data is numpy array, but cpp_integration expects list of [[[x,y,z], [rx,ry,rz]], ...]
    tomogram_raw_data = []
    for particle in test_data:
        pos = particle[:3]  # x, y, z
        orient = particle[3:]  # rx, ry, rz
        tomogram_raw_data.append([pos, orient])
    lattice_assignments = clean_tomo_with_cpp(tomogram_raw_data, test_cleaner)

    # Convert lattice assignments back to list format for comparison
    counts_cpp = [0] * len(test_data)
    for lattice_id, particle_indices in lattice_assignments.items():
        for particle_idx in particle_indices:
            counts_cpp[particle_idx] = lattice_id

    # Use lattice-specific verification
    verify_lattice_assignments(
        python_reference["lattice"], counts_cpp, "Full pipeline", test_data
    )


def calculate_python_reference(
    test_data: np.ndarray, test_cleaner: Cleaner
) -> tuple[dict[str, list[int]], float]:
    """Calculate Python reference results for all filtering stages"""
    logger.info("Calculating Python reference results...")

    start_time = time.time()

    def empty_particle_list() -> list[int]:
        """List of 0s to store particle data for comparison with c++"""
        return [0] * len(base_tomo.all_particles)

    def get_particle_counts(particles) -> list[int]:
        """Helper to get neighbour counts for all particles"""
        counts = empty_particle_list()
        for particle in particles:
            counts[particle.particle_id] = len(particle.neighbours)
        return counts

    def copy_neighbour_sets(particles):
        """Create a copy of neighbour sets for all particles"""
        return {particle: set(particle.neighbours) for particle in particles}

    def restore_neighbour_sets(particles, saved_neighbours):
        """Restore neighbour sets from a saved copy"""
        for particle in particles:
            particle.neighbours = saved_neighbours[particle].copy()

    logger.debug("  Setting up base tomogram with neighbour relationships...")
    base_tomo = setup_test_tomogram()
    all_particles = list(base_tomo.all_particles)

    # Save initial neighbour state - alleviates need to re-instantiate the Tomogram
    initial_neighbours = copy_neighbour_sets(all_particles)

    # Initial neighbour counts (after distance filtering)
    neighbour_counts_initial_python = get_particle_counts(all_particles)

    # Orientation filtering (restore initial state first)
    logger.debug("  Running orientation filtering...")
    restore_neighbour_sets(all_particles, initial_neighbours)
    for particle in all_particles:
        particle.filter_neighbour_orientation(test_cleaner.ori_range, None)
    neighbour_counts_orientation_python = get_particle_counts(all_particles)

    # Curvature filtering (restore initial state first)
    logger.debug("  Running curvature filtering...")
    restore_neighbour_sets(all_particles, initial_neighbours)
    for particle in all_particles:
        particle.filter_curvature(test_cleaner.curv_range)
    neighbour_counts_curvature_python = get_particle_counts(all_particles)

    # Full pipeline (restore initial state first)
    logger.debug("  Running full pipeline...")
    restore_neighbour_sets(all_particles, initial_neighbours)
    for particle in all_particles:
        particle.filter_neighbour_orientation(test_cleaner.ori_range, None)
        particle.filter_curvature(test_cleaner.curv_range)
    neighbour_counts_full_python = get_particle_counts(all_particles)

    # Lattice assignment (need fresh tomogram for autoclean)
    logger.debug("  Running lattice assignment...")
    test_tomo = setup_test_tomogram()
    test_tomo.autoclean()
    lattice_assignments_python = empty_particle_list()
    for particle in test_tomo.all_particles:
        lattice_assignments_python[particle.particle_id] = particle.lattice

    python_time = time.time() - start_time

    return (
        {
            "initial": neighbour_counts_initial_python,
            "orientation": neighbour_counts_orientation_python,
            "curvature": neighbour_counts_curvature_python,
            "full": neighbour_counts_full_python,
            "lattice": lattice_assignments_python,
        },
        python_time,
    )


def verify_counts(
    python_counts: list[int], cpp_counts: list[int], test_name: str
) -> None:
    """Verify that C++ results match Python reference"""
    differences = [
        (i, p, c) for i, (p, c) in enumerate(zip(python_counts, cpp_counts)) if p != c
    ]
    if differences:
        logger.error(f"  FAIL {test_name}: {len(differences)} differences found")
        for i, p, c in differences[:3]:
            logger.error(f"      Particle {i}: Python={p}, C++={c}")
        raise Exception(f"{test_name} failed")
    else:
        logger.info(f"  PASS {test_name}")


def verify_lattice_assignments(
    python_lattices: list[int],
    cpp_lattices: list[int],
    test_name: str,
    test_data: np.ndarray = None,
) -> None:
    """Verify that C++ lattice assignments group particles the same way as Python"""
    # Create mapping from particle index to lattice ID for both implementations
    python_groups = {}
    cpp_groups = {}

    for i, lattice_id in enumerate(python_lattices):
        if lattice_id not in python_groups:
            python_groups[lattice_id] = set()
        python_groups[lattice_id].add(i)

    for i, lattice_id in enumerate(cpp_lattices):
        if lattice_id not in cpp_groups:
            cpp_groups[lattice_id] = set()
        cpp_groups[lattice_id].add(i)

    # Remove lattice 0 (unassigned particles) from comparison
    python_unassigned = python_groups.pop(0, set())
    cpp_unassigned = cpp_groups.pop(0, set())

    # Convert to sets of particle sets for comparison
    python_group_sets = set(frozenset(group) for group in python_groups.values())
    cpp_group_sets = set(frozenset(group) for group in cpp_groups.values())

    logger.debug("      Generating cone plots for debugging...")

    if python_group_sets == cpp_group_sets:
        logger.info(f"PASSED:  {test_name}")
    else:
        logger.warning(f"FAILED:  {test_name}: Lattice groupings differ")
        logger.warning(
            f"   Python: {len(python_group_sets)} lattices, {len(python_unassigned)} unassigned"
        )
        logger.warning(
            f"   C++: {len(cpp_group_sets)} lattices, {len(cpp_unassigned)} unassigned"
        )

        # Find differences with meaningful comparisons
        python_only = python_group_sets - cpp_group_sets
        cpp_only = cpp_group_sets - python_group_sets

        if python_only or cpp_only:
            logger.warning("      Differences found:")

            # Show Python groups that don't exist in C++
            if python_only:
                logger.warning(f"      Python-only lattices ({len(python_only)}):")
                for i, group in enumerate(list(python_only)[:3]):
                    logger.warning(f"        Lattice {i + 1}: {sorted(group)}")

                    # Find closest C++ match for comparison
                    best_match = None
                    best_overlap = 0
                    for cpp_group in cpp_group_sets:
                        overlap = len(group & cpp_group)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = cpp_group

                    if best_match:
                        logger.warning(
                            f"          Closest C++ match: {sorted(best_match)} "
                            f"(overlap: {best_overlap}/{len(group)})"
                        )
                        # Show the actual differences
                        python_only_particles = group - best_match
                        cpp_only_particles = best_match - group
                        if python_only_particles:
                            logger.warning(
                                f"          Python extra particles: {sorted(python_only_particles)}"
                            )
                        if cpp_only_particles:
                            logger.warning(
                                f"          C++ extra particles: {sorted(cpp_only_particles)}"
                            )
                    else:
                        logger.warning("          No similar C++ group found")

            # Show C++ groups that don't exist in Python
            if cpp_only:
                logger.warning(f"      C++-only lattices ({len(cpp_only)}):")
                for i, group in enumerate(list(cpp_only)[:3]):
                    logger.warning(f"        Lattice {i + 1}: {sorted(group)}")

                    # Find closest Python match for comparison
                    best_match = None
                    best_overlap = 0
                    for python_group in python_group_sets:
                        overlap = len(group & python_group)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = python_group

                    if best_match:
                        logger.warning(
                            f"          Closest Python match: {sorted(best_match)} \
                                (overlap: {best_overlap}/{len(group)})"
                        )
                        # Show the actual differences
                        cpp_only_particles = group - best_match
                        python_only_particles = best_match - group
                        if cpp_only_particles:
                            logger.warning(
                                f"          C++ extra particles: \
                                    {sorted(cpp_only_particles)}"
                            )
                        if python_only_particles:
                            logger.warning(
                                f"          Python extra particles: \
                                    {sorted(python_only_particles)}"
                            )
                    else:
                        logger.warning("          No similar Python group found")

        raise Exception(f"{test_name} failed")


def verify_neighbour_relationships(
    c_lib, test_data: np.ndarray, test_cleaner: Cleaner, test_name: str
) -> None:
    """Verify that C++ and Python have identical neighbour relationships at each stage"""
    logger.info(f"TEST {test_name}: Neighbour relationship verification")

    # Get Python neighbour relationships at each stage
    test_tomo = setup_test_tomogram()

    # Stage 1: After distance filtering
    python_neighbours_distance = {}
    for particle in test_tomo.all_particles:
        python_neighbours_distance[particle.particle_id] = {
            p.particle_id for p in particle.neighbours
        }

    # Stage 2: After orientation filtering
    for particle in test_tomo.all_particles:
        particle.filter_neighbour_orientation(test_cleaner.ori_range, None)
    python_neighbours_orientation = {}
    for particle in test_tomo.all_particles:
        python_neighbours_orientation[particle.particle_id] = {
            p.particle_id for p in particle.neighbours
        }

    # Stage 3: After curvature filtering
    test_tomo = setup_test_tomogram()
    for particle in test_tomo.all_particles:
        particle.filter_neighbour_orientation(test_cleaner.ori_range, None)
        particle.filter_curvature(test_cleaner.curv_range)
    python_neighbours_curvature = {}
    for particle in test_tomo.all_particles:
        python_neighbours_curvature[particle.particle_id] = {
            p.particle_id for p in particle.neighbours
        }

    # Get C++ neighbour relationships
    flat_data = [val for particle in test_data for val in particle]
    c_array = (ctypes.c_float * len(flat_data))(*flat_data)
    results_array = (ctypes.c_int * len(test_data))()

    logger.info("  Comparing neighbour relationships...")

    # Compare distance stage
    c_lib.find_neighbours(
        c_array,
        len(test_data),
        test_cleaner.dist_range[0],
        test_cleaner.dist_range[1],
        results_array,
    )
    cpp_counts_distance = [results_array[i] for i in range(len(test_data))]

    # Compare orientation stage
    c_lib.filter_by_orientation(
        c_array,
        len(test_data),
        test_cleaner.ori_range[0],
        test_cleaner.ori_range[1],
        results_array,
    )
    cpp_counts_orientation = [results_array[i] for i in range(len(test_data))]

    # Compare curvature stage
    c_lib.filter_by_curvature(
        c_array,
        len(test_data),
        test_cleaner.curv_range[0],
        test_cleaner.curv_range[1],
        results_array,
    )
    cpp_counts_curvature = [results_array[i] for i in range(len(test_data))]

    # Check if neighbour counts match
    distance_diffs = [
        (i, python_neighbours_distance[i], cpp_counts_distance[i])
        for i in range(len(test_data))
        if len(python_neighbours_distance[i]) != cpp_counts_distance[i]
    ]

    orientation_diffs = [
        (i, len(python_neighbours_orientation[i]), cpp_counts_orientation[i])
        for i in range(len(test_data))
        if len(python_neighbours_orientation[i]) != cpp_counts_orientation[i]
    ]

    curvature_diffs = [
        (i, len(python_neighbours_curvature[i]), cpp_counts_curvature[i])
        for i in range(len(test_data))
        if len(python_neighbours_curvature[i]) != cpp_counts_curvature[i]
    ]

    if distance_diffs:
        logger.error(
            f"  FAIL Distance stage: {len(distance_diffs)} particles have "
            f"different neighbour counts"
        )
        for i, python_count, cpp_count in distance_diffs[:5]:
            logger.error(
                f"      Particle {i}: Python={python_count} neighbours, C++={cpp_count} neighbours"
            )
        raise Exception(f"{test_name} failed at distance stage")

    if orientation_diffs:
        logger.error(
            f"   Orientation stage: {len(orientation_diffs)} \
                particles have different neighbour counts"
        )
        for i, python_count, cpp_count in orientation_diffs[:5]:
            logger.error(
                f"      Particle {i}: Python={python_count} neighbours, C++={cpp_count} neighbours"
            )
        raise Exception(f"{test_name} failed at orientation stage")

    if curvature_diffs:
        logger.error(
            f"  FAIL Curvature stage: {len(curvature_diffs)} particles have "
            f"different neighbour counts"
        )
        for i, python_count, cpp_count in curvature_diffs[:5]:
            logger.error(
                f"      Particle {i}: Python={python_count} neighbours, C++={cpp_count} neighbours"
            )
        raise Exception(f"{test_name} failed at curvature stage")

    logger.info("  PASS All neighbour counts match between Python and C++")
    logger.info(
        "  PASS The differences are indeed only in the lattice assignment phase"
    )


def compare_results(
    cpp_distance_time: float,
    cpp_orientation_time: float,
    cpp_curvature_time: float,
    cpp_full_time: float,
    python_time: float,
) -> None:
    """Display performance metrics"""
    speedup_distance = (
        python_time / cpp_distance_time if cpp_distance_time > 0 else float("inf")
    )
    speedup_orientation = (
        python_time / cpp_orientation_time if cpp_orientation_time > 0 else float("inf")
    )
    speedup_curvature = (
        python_time / cpp_curvature_time if cpp_curvature_time > 0 else float("inf")
    )
    speedup_full = python_time / cpp_full_time if cpp_full_time > 0 else float("inf")

    logger.info("\nPERFORMANCE RESULTS:")
    logger.info(
        f"Python: {python_time:.4f}s \
            | C++ Distance: {cpp_distance_time:.4f}s ({speedup_distance:.1f}x) \
            | Orientation: {cpp_orientation_time:.4f}s ({speedup_orientation:.1f}x) \
            | Curvature: {cpp_curvature_time:.4f}s ({speedup_curvature:.1f}x) \
            | Full: {cpp_full_time:.4f}s ({speedup_full:.1f}x)"
    )


def create_cone_plot_from_lattices(
    test_data: np.ndarray, lattice_assignments: list[int], title: str, filename: str
) -> None:
    """Create a cone plot from lattice assignments for debugging"""

    # Create tomogram and particles
    test_tomo = create_test_tomogram()

    for particle in test_tomo.all_particles:
        particle.set_lattice(lattice_assignments[particle.particle_id])

    # Generate cone plot
    fig = test_tomo.plot_all_lattices(cone_size=3)
    fig.write_html(filename)
    logger.info(f"Saved cone plot to: {filename}")
    return fig
