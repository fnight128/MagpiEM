import ctypes
import pathlib

current_dir = pathlib.Path().absolute()
libname = current_dir / "processing.dll"

print("loading library")

assert libname.exists(), "File does not exist"
c_lib = ctypes.CDLL(str(libname))

arg_types = [ctypes.c_int, ctypes.c_float]
return_type = ctypes.c_float

c_lib.cmult.argtypes = arg_types
c_lib.cmult.restype = return_type

print("arg types", arg_types)
print("return type", return_type)

test_cases = [
    (5, 2.5),
    (10, 3.7),
    (0, 4.2),
    (-3, 2.0),
    (7, 0.0)
]

for int_val, float_val in test_cases:
    print(f"\nCalling cmult({int_val}, {float_val})")
    result = c_lib.cmult(int_val, float_val)

    print(f"Result: {result}")
    print(f"Expected: {int_val * float_val}")
