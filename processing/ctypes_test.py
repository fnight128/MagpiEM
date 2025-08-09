import ctypes
import pathlib

current_dir = pathlib.Path().absolute()
libname = current_dir / "processing.dll"

print("loading library")

assert libname.exists(), "File does not exist"
c_lib = ctypes.CDLL(str(libname))

arg_types = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
# Output is returned into the array given as the final argument
return_type = None 

c_lib.clean_particles.argtypes = arg_types
c_lib.clean_particles.restype = return_type

print("arg types", arg_types)
print("return type", return_type)

test_cases = [
    # [x, y, z, rx, ry, rz]
    [
        [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
        [7.0, 8.0, 9.0, 0.7, 0.8, 0.9],
        [10.0, 11.0, 12.0, 1.0, 1.1, 1.2],
    ],
]

for i, points_data in enumerate(test_cases):
    print(f"\n{'='*60}")
    print(f"Test case {i+1}: {len(points_data)} oriented points")
    
    flat_data = [i for s in points_data for i in s]
    
    c_array = (ctypes.c_float * len(flat_data))(*flat_data)

    results_array = (ctypes.c_int * len(points_data))()

    print(f"\nCleaning (data, {len(points_data)}, results)")
    c_lib.clean_particles(c_array, len(points_data), results_array)
    
    results = [results_array[i] for i in range(len(points_data))]
    print(f"Results: {results}")
    
    expected = [int(sum(point_data[:3])) for point_data in points_data]
    print(f"Expected: {expected}")
