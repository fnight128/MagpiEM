import ctypes
import pathlib
import numpy as np
import time
from magpiem.read_write import read_emc_mat, read_single_tomogram
from magpiem.classes import Tomogram, Particle, Cleaner

current_dir = pathlib.Path().absolute()
libname = current_dir / "processing.dll"

print("loading library")

assert libname.exists(), "File does not exist"
print("DLL exists")
c_lib = ctypes.CDLL(str(libname))

test_data_file = "test_data.mat"
test_tomo_name = "wt2nd_4004_2"
test_cleaner = Cleaner.from_user_params(2, 3, 10, 60, 40, 10, 20, 90, 20)

# Time Python calculation
print("\n" + "="*60)
print("TIMING COMPARISON: Python vs C++")
print("="*60)

# Python timing
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

arg_types = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_int),
]
# Output is returned into the array given as the final argument
return_type = None

c_lib.clean_particles.argtypes = arg_types
c_lib.clean_particles.restype = return_type

print("arg types", arg_types)
print("return type", return_type)

min_dist, max_dist = test_cleaner.dist_range

test_data = np.array(read_emc_mat(test_data_file)[test_tomo_name], dtype=float)
test_data = test_data[:,[10,11,12,22,23,24]]

print(f"\n{'='*60}")
print(f"Particles: {len(test_data)}, Range: {min_dist}-{max_dist} units")

flat_data = [val for particle in test_data for val in particle]

c_array = (ctypes.c_float * len(flat_data))(*flat_data)
results_array = (ctypes.c_int * len(test_data))()

# C++ timing
print("\nRunning C++ neighbor calculation...")
start_time = time.time()

print(f"\nCalling clean_particles(data, {len(test_data)}, {min_dist}, {max_dist}, results)")
c_lib.clean_particles(c_array, len(test_data), min_dist, max_dist, results_array)

cpp_time = time.time() - start_time
print(f"C++ calculation time: {cpp_time:.4f} seconds")

neighbour_counts_cpp = [results_array[i] for i in range(len(test_data))]

# Performance comparison
speedup = python_time / cpp_time if cpp_time > 0 else float('inf')
print(f"\n{'='*60}")
print(f"PERFORMANCE RESULTS:")
print(f"Python time: {python_time:.4f} seconds")
print(f"C++ time:    {cpp_time:.4f} seconds")
print(f"Speedup:     {speedup:.2f}x faster")
print(f"{'='*60}")

# Verify neighbor counts match
assert neighbour_counts_python == neighbour_counts_cpp, "Neighbour count failed"

print("All tests run successfully")
