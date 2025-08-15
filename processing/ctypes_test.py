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

print(f"Imported Cleaner from: {Cleaner.__module__}")
print(f"Cleaner methods: {[m for m in dir(Cleaner) if not m.startswith('_')]}")

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
        ("min_neighbors", ctypes.c_int)
    ]

libname = current_dir / "processing.dll"

print("loading library")

assert libname.exists(), "File does not exist"
print("DLL exists")
c_lib = ctypes.CDLL(str(libname))

test_data_file = "test_data.mat"
test_tomo_name = "wt2nd_4004_2"
# Cleaning parameters for test dataset
test_cleaner = Cleaner.from_user_params(2.0, 3, 10, 60.0, 40.0, 10.0, 20.0, 90.0, 20.0)

# Extract all parameters from the cleaner object
min_dist, max_dist = test_cleaner.dist_range
min_ori, max_ori = test_cleaner.ori_range
min_curv, max_curv = test_cleaner.curv_range
min_lattice_size = test_cleaner.min_lattice_size
min_neighbors = test_cleaner.min_neighbours

print(f"Cleaning Parameters:")
print(f"  Distance range: {min_dist:.2f} - {max_dist:.2f}")
print(f"  Orientation range: {min_ori:.2f} - {max_ori:.2f}")
print(f"  Curvature range: {min_curv:.2f} - {max_curv:.2f}")
print(f"  Min lattice size: {min_lattice_size}")
print(f"  Min neighbors: {min_neighbors}")

test_data = np.array(read_emc_mat(test_data_file)[test_tomo_name], dtype=float)
test_data = test_data[:,[10,11,12,22,23,24]]

print(f"\n{'='*60}")
print(f"Particles: {len(test_data)}, Range: {min_dist:.2f}-{max_dist:.2f} units")

flat_data = [val for particle in test_data for val in particle]
c_array = (ctypes.c_float * len(flat_data))(*flat_data)
results_array = (ctypes.c_int * len(test_data))()

# Set up function signatures
c_lib.find_neighbors.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
c_lib.find_neighbors.restype = None

c_lib.filter_by_orientation.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
c_lib.filter_by_orientation.restype = None

c_lib.clean_particles.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(CleanParams), ctypes.POINTER(ctypes.c_int)]
c_lib.clean_particles.restype = None

# Test 1: Distance-only neighbor finding
print("\n" + "="*60)
print("TEST 1: Distance-only neighbor finding")
print("="*60)

print("\nRunning C++ distance-only neighbor finding...")
start_time = time.time()
c_lib.find_neighbors(c_array, len(test_data), min_dist, max_dist, results_array)
cpp_distance_time = time.time() - start_time
print(f"C++ distance-only time: {cpp_distance_time:.4f} seconds")

distance_counts_cpp = [results_array[i] for i in range(len(test_data))]
print(f"First 10 distance-only counts: {distance_counts_cpp[:10]}")

# Test 2: Full pipeline (distance + orientation)
print("\n" + "="*60)
print("TEST 2: Full pipeline (distance + orientation)")
print("="*60)

params = CleanParams(
    min_distance=min_dist,
    max_distance=max_dist,
    min_orientation=min_ori,
    max_orientation=max_ori,
    min_curvature=min_curv,
    max_curvature=max_curv,
    min_lattice_size=min_lattice_size,
    min_neighbors=min_neighbors
)

print("\nRunning C++ full pipeline...")
start_time = time.time()
c_lib.clean_particles(c_array, len(test_data), ctypes.byref(params), results_array)
cpp_full_time = time.time() - start_time
print(f"C++ full pipeline time: {cpp_full_time:.4f} seconds")

full_counts_cpp = [results_array[i] for i in range(len(test_data))]
print(f"First 10 full pipeline counts: {full_counts_cpp[:10]}")

# Test 3: Python neighbour count
print("\n" + "="*60)
print("TEST 3: Python neighbour count")
print("="*60)

print("\nRunning Python neighbor calculation...")
start_time = time.time()

test_tomo = read_single_tomogram(test_data_file, test_tomo_name)
test_tomo.set_clean_params(test_cleaner)
test_tomo.find_particle_neighbours(test_cleaner.dist_range)
neighbour_counts_initial_python = [0] * len(test_tomo.all_particles)
for particle in test_tomo.all_particles:
    neighbour_counts_initial_python[particle.particle_id] = len(particle.neighbours)

# each filtration step must be run in a separate loop, as particles remove neighbours from other particles during filtration
neighbour_counts_orientation_python = [0] * len(test_tomo.all_particles)
for particle in test_tomo.all_particles:
    particle.filter_neighbour_orientation(test_cleaner.ori_range, None)
    neighbour_counts_orientation_python[particle.particle_id] = len(particle.neighbours)

test_tomo.find_particle_neighbours(test_cleaner.dist_range)
neighbour_counts_python = [0] * len(test_tomo.all_particles)
for particle in test_tomo.all_particles:
    neighbour_counts_python[particle.particle_id] = len(particle.neighbours)


python_time = time.time() - start_time
print(f"Python calculation time: {python_time:.4f} seconds")
print(f"First 10 Python initial counts: {neighbour_counts_initial_python[:10]}")
print(f"First 10 Python orientation-filtered counts: {neighbour_counts_orientation_python[:10]}")

# Performance comparison
speedup_distance = python_time / cpp_distance_time if cpp_distance_time > 0 else float('inf')
speedup_full = python_time / cpp_full_time if cpp_full_time > 0 else float('inf')
print(f"\n{'='*60}")
print(f"PERFORMANCE RESULTS:")
print(f"Python time: {python_time:.4f} seconds")
print(f"C++ distance-only time: {cpp_distance_time:.4f} seconds")
print(f"C++ full pipeline time: {cpp_full_time:.4f} seconds")
print(f"Speedup (distance-only): {speedup_distance:.2f}x faster")
print(f"Speedup (full pipeline): {speedup_full:.2f}x faster")
print(f"{'='*60}")

# Verify distance-only results match Python
distance_differences = [(i, p, c) for i, (p, c) in enumerate(zip(neighbour_counts_initial_python, distance_counts_cpp)) if p != c]
if distance_differences:
    print(f"\nFound {len(distance_differences)} differences in distance-only neighbor counts:")
    for i, p, c in distance_differences[:10]:
        print(f"  Particle {i}: Python={p}, C++={c}")
    raise Exception("Distance-only neighbour count failed")
else:
    print("\n✓ Distance-only neighbor counts match!")

# Verify orientation filtering results match Python
orientation_differences = [(i, p, c) for i, (p, c) in enumerate(zip(neighbour_counts_orientation_python, full_counts_cpp)) if p != c]
if orientation_differences:
    print(f"\nFound {len(orientation_differences)} differences in orientation-filtered neighbor counts:")
    for i, p, c in orientation_differences[:10]:
        print(f"  Particle {i}: Python={p}, C++={c}")
    raise Exception("Orientation-filtered neighbour count failed")
else:
    print("\n✓ Orientation-filtered neighbor counts match!")

print("All tests run successfully")
