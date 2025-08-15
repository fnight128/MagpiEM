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

print("\nRunning Python neighbor calculation...")
start_time = time.time()

test_tomo = read_single_tomogram(test_data_file, test_tomo_name)
test_tomo.set_clean_params(test_cleaner)
test_tomo.find_particle_neighbours(test_cleaner.dist_range)
neighbour_counts_python = [0] * len(test_tomo.all_particles)
for particle in test_tomo.all_particles:
    neighbour_counts_python[particle.particle_id] = len(particle.neighbours)

python_time = time.time() - start_time
print(f"Python calculation time: {python_time:.4f} seconds")

# Set up function signature
arg_types = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(CleanParams),
    ctypes.POINTER(ctypes.c_int),
]
return_type = None

c_lib.clean_particles.argtypes = arg_types
c_lib.clean_particles.restype = return_type

print("arg types", arg_types)
print("return type", return_type)

test_data = np.array(read_emc_mat(test_data_file)[test_tomo_name], dtype=float)
test_data = test_data[:,[10,11,12,22,23,24]]

print(f"\n{'='*60}")
print(f"Particles: {len(test_data)}, Range: {min_dist:.2f}-{max_dist:.2f} units")

flat_data = [val for particle in test_data for val in particle]

c_array = (ctypes.c_float * len(flat_data))(*flat_data)
results_array = (ctypes.c_int * len(test_data))()

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

print("\nRunning C++ neighbor calculation...")
start_time = time.time()

print(f"\nCalling clean_particles(data, {len(test_data)}, params, results)")
c_lib.clean_particles(c_array, len(test_data), ctypes.byref(params), results_array)

cpp_time = time.time() - start_time
print(f"C++ calculation time: {cpp_time:.4f} seconds")

neighbour_counts_cpp = [results_array[i] for i in range(len(test_data))]

speedup = python_time / cpp_time if cpp_time > 0 else float('inf')
print(f"\n{'='*60}")
print(f"PERFORMANCE RESULTS:")
print(f"Python time: {python_time:.4f} seconds")
print(f"C++ time:    {cpp_time:.4f} seconds")
print(f"Speedup:     {speedup:.2f}x faster")
print(f"{'='*60}")

# Check if there are any differences
differences = [(i, p, c) for i, (p, c) in enumerate(zip(neighbour_counts_python, neighbour_counts_cpp)) if p != c]
if differences:
    print(f"\nFound {len(differences)} differences in neighbor counts:")
    for i, p, c in differences[:10]:  # Show first 10 differences
        print(f"  Particle {i}: Python={p}, C++={c}")
    raise Exception("Neighbour count failed")
    
print("All tests run successfully")
