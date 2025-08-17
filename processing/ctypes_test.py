import ctypes
import pathlib
import numpy as np
import time
import sys

# Add the parent directory to the path to ensure we use the local magpiem module
current_dir = pathlib.Path().absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from magpiem.read_write import read_emc_mat, read_single_tomogram
from magpiem.classes import Tomogram, Particle, Cleaner

TEST_DATA_FILE = "test_data.mat"
TEST_TOMO_NAME = "wt2nd_4004_2"
# appropriate values for test tomogram
TEST_CLEANER_VALUES = [2.0, 3, 10, 60.0, 40.0, 10.0, 20.0, 90.0, 20.0]

# Define the CleanParams struct for ctypes
class CleanParams(ctypes.Structure):
    _fields_ = [
        ("min_distance", ctypes.c_float),
        ("max_distance", ctypes.c_float),
        ("min_orientation", ctypes.c_float),
        ("max_orientation", ctypes.c_float),
        ("min_curvature", ctypes.c_float),
        ("max_curvature", ctypes.c_float),
        ("min_lattice_size", ctypes.c_int),
        ("min_neighbours", ctypes.c_int)
    ]

def setup_cpp_library() -> ctypes.CDLL:
    """Load and configure the C++ library"""
    libname = current_dir / "processing.dll"
    print("loading library")
    
    assert libname.exists(), "File does not exist"
    print("DLL exists")
    c_lib = ctypes.CDLL(str(libname))
    
    # Set up function signatures
    c_lib.find_neighbours.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
    c_lib.find_neighbours.restype = None
    
    c_lib.filter_by_orientation.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
    c_lib.filter_by_orientation.restype = None
    
    c_lib.filter_by_curvature.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
    c_lib.filter_by_curvature.restype = None
    
    c_lib.assign_lattices.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    c_lib.assign_lattices.restype = None
    
    c_lib.clean_particles.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(CleanParams), ctypes.POINTER(ctypes.c_int)]
    c_lib.clean_particles.restype = None
    
    return c_lib

def load_particle_data() -> np.ndarray:
    """Load particle data from the test file"""
    test_data = np.array(read_emc_mat(TEST_DATA_FILE)[TEST_TOMO_NAME], dtype=float)
    test_data = test_data[:,[10,11,12,22,23,24]]  # Extract position and orientation columns
    return test_data

def load_test_data() -> tuple[np.ndarray, Cleaner, tuple[float, float, float, float, float, float, int, int]]:
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
    
    print(f"Testing with {len(test_data)} particles")
    
    return test_data, test_cleaner, (min_dist, max_dist, min_ori, max_ori, min_curv, max_curv, min_lattice_size, min_neighbours)

def create_test_tomogram() -> Tomogram:
    return read_single_tomogram(TEST_DATA_FILE, TEST_TOMO_NAME)

def setup_test_tomogram(test_cleaner: Cleaner) -> Tomogram:
    """Create and setup a test tomogram with the given cleaner parameters"""
    test_tomo = create_test_tomogram()
    test_tomo.set_clean_params(test_cleaner)
    test_tomo.find_particle_neighbours(test_cleaner.dist_range)
    return test_tomo

def run_cpp_test(c_lib: ctypes.CDLL, test_data: np.ndarray, test_name: str, python_reference: list[int], 
                test_func, *args) -> tuple[list[int], float]:
    """Generic function to run C++ tests with common setup and validation"""
    print(f"TEST {test_name}")
    
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

def test_cpp_distance_only(c_lib: ctypes.CDLL, test_data: np.ndarray, min_dist: float, max_dist: float, python_reference: list[int]) -> tuple[list[int], float]:
    """Test C++ distance-only neighbour finding"""
    def distance_test(c_lib, c_array, num_particles, results_array, min_dist, max_dist):
        c_lib.find_neighbours(c_array, num_particles, min_dist, max_dist, results_array)
    
    return run_cpp_test(c_lib, test_data, "1: Distance-only neighbour finding", python_reference, 
                       distance_test, min_dist, max_dist)

def test_cpp_orientation_only(c_lib: ctypes.CDLL, test_data: np.ndarray, min_dist: float, max_dist: float, min_ori: float, max_ori: float, python_reference: list[int]) -> tuple[list[int], float]:
    """Test C++ orientation filtering alone (after distance filtering)"""
    def orientation_test(c_lib, c_array, num_particles, results_array, min_dist, max_dist, min_ori, max_ori):
        c_lib.find_neighbours(c_array, num_particles, min_dist, max_dist, results_array)
        c_lib.filter_by_orientation(c_array, num_particles, min_ori, max_ori, results_array)
    
    return run_cpp_test(c_lib, test_data, "2: Orientation filtering", python_reference, 
                       orientation_test, min_dist, max_dist, min_ori, max_ori)

def test_cpp_curvature_only(c_lib: ctypes.CDLL, test_data: np.ndarray, min_dist: float, max_dist: float, min_curv: float, max_curv: float, python_reference: list[int]) -> tuple[list[int], float]:
    """Test C++ curvature filtering alone (after distance filtering)"""
    def curvature_test(c_lib, c_array, num_particles, results_array, min_dist, max_dist, min_curv, max_curv):
        c_lib.find_neighbours(c_array, num_particles, min_dist, max_dist, results_array)
        c_lib.filter_by_curvature(c_array, num_particles, min_curv, max_curv, results_array)
    
    return run_cpp_test(c_lib, test_data, "3: Curvature filtering", python_reference, 
                       curvature_test, min_dist, max_dist, min_curv, max_curv)

def test_cpp_full_pipeline(c_lib: ctypes.CDLL, test_data: np.ndarray, params: CleanParams, python_reference: list[int]) -> tuple[list[int], float]:
    """Test C++ full pipeline (distance + orientation + curvature + lattice assignment)"""
    def full_pipeline_test(c_lib, c_array, num_particles, results_array, params):
        c_lib.clean_particles(c_array, num_particles, ctypes.byref(params), results_array)
    
    # Use custom validation for lattice assignments, as lattice IDs are arbitrarily assigned, and likely to differ between implementations
    print("TEST 4: Full pipeline")
    
    flat_data = [val for particle in test_data for val in particle]
    c_array = (ctypes.c_float * len(flat_data))(*flat_data)
    results_array = (ctypes.c_int * len(test_data))()
    
    start_time = time.time()
    full_pipeline_test(c_lib, c_array, len(test_data), results_array, params)
    test_time = time.time() - start_time
    
    counts_cpp = [results_array[i] for i in range(len(test_data))]
    
    # Use lattice-specific verification
    verify_lattice_assignments(python_reference, counts_cpp, "Full pipeline", test_data)
    
    return counts_cpp, test_time



def calculate_python_reference(test_data: np.ndarray, test_cleaner: Cleaner) -> tuple[dict[str, list[int]], float]:
    """Calculate Python reference results for all filtering stages"""
    print("Calculating Python reference results...")
    
    start_time = time.time()

    def empty_particle_list() -> list[int]:
        """List of 0s to store particle data for comparison with c++"""
        return [0] * len(test_tomo.all_particles)
    
    test_tomo = setup_test_tomogram(test_cleaner)
    neighbour_counts_initial_python = empty_particle_list()
    for particle in test_tomo.all_particles:
        neighbour_counts_initial_python[particle.particle_id] = len(particle.neighbours)
    
    # each filtration step must be run in a separate loop, as particles remove neighbours from other particles during filtration
    neighbour_counts_orientation_python = empty_particle_list()
    for particle in test_tomo.all_particles:
        particle.filter_neighbour_orientation(test_cleaner.ori_range, None)
        neighbour_counts_orientation_python[particle.particle_id] = len(particle.neighbours)

    test_tomo = setup_test_tomogram(test_cleaner)
    neighbour_counts_curvature_python = empty_particle_list()
    for particle in test_tomo.all_particles:
        particle.filter_curvature(test_cleaner.curv_range)
        neighbour_counts_curvature_python[particle.particle_id] = len(particle.neighbours)
    
    # Test full pipeline (distance + orientation + curvature)
    test_tomo = setup_test_tomogram(test_cleaner)
    neighbour_counts_full_python = empty_particle_list()
    for particle in test_tomo.all_particles:
        particle.filter_neighbour_orientation(test_cleaner.ori_range, None)
        particle.filter_curvature(test_cleaner.curv_range)
        neighbour_counts_full_python[particle.particle_id] = len(particle.neighbours)
    
    # Test lattice assignment
    lattice_assignments_python = empty_particle_list()
    test_tomo = setup_test_tomogram(test_cleaner)
    test_tomo.autoclean()
    
    for particle in test_tomo.all_particles:
        lattice_assignments_python[particle.particle_id] = particle.lattice
    
    python_time = time.time() - start_time
    
    return ({"initial": neighbour_counts_initial_python, "orientation": neighbour_counts_orientation_python, 
            "curvature": neighbour_counts_curvature_python, "full": neighbour_counts_full_python, 
            "lattice": lattice_assignments_python}, python_time)

def verify_counts(python_counts: list[int], cpp_counts: list[int], test_name: str) -> None:
    """Verify that C++ results match Python reference"""
    differences = [(i, p, c) for i, (p, c) in enumerate(zip(python_counts, cpp_counts)) if p != c]
    if differences:
        print(f"  ❌ {test_name}: {len(differences)} differences found")
        for i, p, c in differences[:3]:
            print(f"      Particle {i}: Python={p}, C++={c}")
        raise Exception(f"{test_name} failed")
    else:
        print(f"  ✓ {test_name}")

def verify_lattice_assignments(python_lattices: list[int], cpp_lattices: list[int], test_name: str, test_data: np.ndarray = None) -> None:
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
    
    if python_group_sets == cpp_group_sets:
        print(f"  ✓ {test_name}")
    else:
        print(f"  ❌ {test_name}: Lattice groupings differ")
        print(f"      Python: {len(python_group_sets)} lattices, {len(python_unassigned)} unassigned")
        print(f"      C++: {len(cpp_group_sets)} lattices, {len(cpp_unassigned)} unassigned")
        
        # Generate cone plots for debugging if test_data is provided
        if test_data is not None:
            print(f"      Generating cone plots for debugging...")
            create_cone_plot_from_lattices(test_data, cpp_lattices, "C++ Lattice Assignments", "cpp_lattices.html")
            create_cone_plot_from_lattices(test_data, python_lattices, "Python Lattice Assignments", "python_lattices.html")
        
        # Find differences with meaningful comparisons
        python_only = python_group_sets - cpp_group_sets
        cpp_only = cpp_group_sets - python_group_sets
        
        if python_only or cpp_only:
            print(f"      Differences found:")
            
            # Show Python groups that don't exist in C++
            if python_only:
                print(f"      Python-only lattices ({len(python_only)}):")
                for i, group in enumerate(list(python_only)[:3]):
                    print(f"        Lattice {i+1}: {sorted(group)}")
                    
                    # Find closest C++ match for comparison
                    best_match = None
                    best_overlap = 0
                    for cpp_group in cpp_group_sets:
                        overlap = len(group & cpp_group)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = cpp_group
                    
                    if best_match:
                        print(f"          Closest C++ match: {sorted(best_match)} (overlap: {best_overlap}/{len(group)})")
                        # Show the actual differences
                        python_only_particles = group - best_match
                        cpp_only_particles = best_match - group
                        if python_only_particles:
                            print(f"          Python extra particles: {sorted(python_only_particles)}")
                        if cpp_only_particles:
                            print(f"          C++ extra particles: {sorted(cpp_only_particles)}")
                    else:
                        print(f"          No similar C++ group found")
            
            # Show C++ groups that don't exist in Python
            if cpp_only:
                print(f"      C++-only lattices ({len(cpp_only)}):")
                for i, group in enumerate(list(cpp_only)[:3]):
                    print(f"        Lattice {i+1}: {sorted(group)}")
                    
                    # Find closest Python match for comparison
                    best_match = None
                    best_overlap = 0
                    for python_group in python_group_sets:
                        overlap = len(group & python_group)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = python_group
                    
                    if best_match:
                        print(f"          Closest Python match: {sorted(best_match)} (overlap: {best_overlap}/{len(group)})")
                        # Show the actual differences
                        cpp_only_particles = group - best_match
                        python_only_particles = best_match - group
                        if cpp_only_particles:
                            print(f"          C++ extra particles: {sorted(cpp_only_particles)}")
                        if python_only_particles:
                            print(f"          Python extra particles: {sorted(python_only_particles)}")
                    else:
                        print(f"          No similar Python group found")
        
        raise Exception(f"{test_name} failed")

def compare_results(cpp_distance_time: float, cpp_orientation_time: float, cpp_curvature_time: float, cpp_full_time: float, python_time: float) -> None:
    """Display performance metrics"""
    speedup_distance = python_time / cpp_distance_time if cpp_distance_time > 0 else float('inf')
    speedup_orientation = python_time / cpp_orientation_time if cpp_orientation_time > 0 else float('inf')
    speedup_curvature = python_time / cpp_curvature_time if cpp_curvature_time > 0 else float('inf')
    speedup_full = python_time / cpp_full_time if cpp_full_time > 0 else float('inf')
    
    print(f"\nPERFORMANCE RESULTS:")
    print(f"Python: {python_time:.4f}s | C++ Distance: {cpp_distance_time:.4f}s ({speedup_distance:.1f}x) | Orientation: {cpp_orientation_time:.4f}s ({speedup_orientation:.1f}x) | Curvature: {cpp_curvature_time:.4f}s ({speedup_curvature:.1f}x) | Full: {cpp_full_time:.4f}s ({speedup_full:.1f}x)")

def create_cone_plot_from_lattices(test_data: np.ndarray, lattice_assignments: list[int], title: str, filename: str) -> None:
    """Create a cone plot from lattice assignments for debugging"""
    from magpiem.classes import Tomogram, Particle
    
    # Create tomogram and particles
    test_tomo = create_test_tomogram()
    
    for particle in test_tomo.all_particles:
        particle.set_lattice(lattice_assignments[particle.particle_id])
    test_tomo.generate_lattice_dfs()

    # Generate cone plot
    fig = test_tomo.plot_all_lattices(cone_size=10)
    fig.write_html(filename)
    print(f"Saved cone plot to: {filename}")

def main() -> None:
    """Main test function"""
    print("Running C++ vs Python particle filtering tests...")
    
    # Setup
    c_lib = setup_cpp_library()
    test_data, test_cleaner, params_tuple = load_test_data()
    min_dist, max_dist, min_ori, max_ori, min_curv, max_curv, min_lattice_size, min_neighbours = params_tuple
    
    # Create CleanParams struct
    params = CleanParams(
        min_distance=min_dist,
        max_distance=max_dist,
        min_orientation=min_ori,
        max_orientation=max_ori,
        min_curvature=min_curv,
        max_curvature=max_curv,
        min_lattice_size=min_lattice_size,
        min_neighbours=min_neighbours
    )

    # Calculate Python reference results first
    python_reference, python_time = calculate_python_reference(test_data, test_cleaner)
    
    # Run C++ tests and validate against Python reference
    distance_counts_cpp, cpp_distance_time = test_cpp_distance_only(c_lib, test_data, min_dist, max_dist, python_reference["initial"])
    orientation_counts_cpp, cpp_orientation_time = test_cpp_orientation_only(c_lib, test_data, min_dist, max_dist, min_ori, max_ori, python_reference["orientation"])
    curvature_counts_cpp, cpp_curvature_time = test_cpp_curvature_only(c_lib, test_data, min_dist, max_dist, min_curv, max_curv, python_reference["curvature"])
    full_counts_cpp, cpp_full_time = test_cpp_full_pipeline(c_lib, test_data, params, python_reference["lattice"])
    
    # Display performance results
    compare_results(cpp_distance_time, cpp_orientation_time, cpp_curvature_time, cpp_full_time, python_time)
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    main()
