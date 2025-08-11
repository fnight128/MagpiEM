import ctypes
import pathlib
from magpiem.read_write import read_emc_mat

current_dir = pathlib.Path().absolute()
libname = current_dir / "processing.dll"

print("loading library")

assert libname.exists(), "File does not exist"
c_lib = ctypes.CDLL(str(libname))

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

test_data = read_emc_mat("test_data.mat")
print(test_data)

test_cases = [
    # Test case: particles and distance parameters
    {
        "particles": [
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
            [3.0, 0.0, 0.0, 0.4, 0.5, 0.6],
            [8.0, 0.0, 0.0, 0.7, 0.8, 0.9],
            [15.0, 0.0, 0.0, 1.0, 1.1, 1.2],
            [2.0, 2.0, 0.0, 1.3, 1.4, 1.5],
        ],
        "min_distance": 2.0,
        "max_distance": 10.0,
    }
]

for i, test_case in enumerate(test_cases):
    particles_data = test_case["particles"]
    min_dist = test_case["min_distance"]
    max_dist = test_case["max_distance"]

    print(f"\n{'='*60}")
    print(f"Test {i+1}")
    print(f"Particles: {len(particles_data)}, Range: {min_dist}-{max_dist} units")

    flat_data = [val for particle in particles_data for val in particle]

    c_array = (ctypes.c_float * len(flat_data))(*flat_data)
    results_array = (ctypes.c_int * len(particles_data))()

    results = [results_array[i] for i in range(len(particles_data))]
    print(f"\nNeighbor counts: {results}")
