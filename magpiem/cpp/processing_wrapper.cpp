#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <cstring>
#include <cstdarg>

static PyModuleDef processing_module = {
    PyModuleDef_HEAD_INIT,
    "processing_cpp",
    "C++ processing functions for MagpiEM",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_processing_cpp(void) {
    return PyModule_Create(&processing_module);
}

extern "C" {
    void clean_particles(float* data, int num_points, CleanParams* params, int* results);
    void find_neighbours(float* data, int num_points, float min_distance, float max_distance, int* results);
    void filter_by_orientation(float* data, int num_points, float min_orientation, float max_orientation, int* results);
    void filter_by_curvature(float* data, int num_points, float min_curvature, float max_curvature, int* results);
    void assign_lattices(float* data, int num_points, unsigned int min_neighbours, unsigned int min_lattice_size, int* results);
    void get_cleaned_neighbours(float* data, int num_points, CleanParams* params, int* offsets, int* neighbours_out);
    void set_log_level(int level);
}

#include "processing.cpp"
