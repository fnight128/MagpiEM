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

// Include the original processing.cpp implementation
#include "processing.cpp"

// Python function wrappers
static PyObject* py_find_neighbours(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int num_particles;
    float min_distance, max_distance;
    
    if (!PyArg_ParseTuple(args, "Oiff", &data_obj, &num_particles, &min_distance, &max_distance)) {
        return NULL;
    }
    
    // Convert Python list/array to C array
    float* data = new float[num_particles * 6];
    for (int i = 0; i < num_particles * 6; i++) {
        PyObject* item = PyList_GetItem(data_obj, i);
        data[i] = (float)PyFloat_AsDouble(item);
    }
    
    int* results = new int[num_particles];
    
    // Call C++ function
    find_neighbours(data, num_particles, min_distance, max_distance, results);
    
    // Convert results back to Python
    PyObject* results_list = PyList_New(num_particles);
    for (int i = 0; i < num_particles; i++) {
        PyList_SetItem(results_list, i, PyLong_FromLong(results[i]));
    }
    
    delete[] data;
    delete[] results;
    
    return results_list;
}

static PyObject* py_filter_by_orientation(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int num_particles;
    float min_orientation, max_orientation;
    int allow_flips_int = 0;
    
    if (!PyArg_ParseTuple(args, "Oiff|i", &data_obj, &num_particles, &min_orientation, &max_orientation, &allow_flips_int)) {
        return NULL;
    }
    
    bool allow_flips = (allow_flips_int != 0);
    
    float* data = new float[num_particles * 6];
    for (int i = 0; i < num_particles * 6; i++) {
        PyObject* item = PyList_GetItem(data_obj, i);
        data[i] = (float)PyFloat_AsDouble(item);
    }
    
    int* results = new int[num_particles];
    
    filter_by_orientation(data, num_particles, min_orientation, max_orientation, allow_flips, results);
    
    PyObject* results_list = PyList_New(num_particles);
    for (int i = 0; i < num_particles; i++) {
        PyList_SetItem(results_list, i, PyLong_FromLong(results[i]));
    }
    
    delete[] data;
    delete[] results;
    
    return results_list;
}

static PyObject* py_filter_by_curvature(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int num_particles;
    float min_curvature, max_curvature;
    
    if (!PyArg_ParseTuple(args, "Oiff", &data_obj, &num_particles, &min_curvature, &max_curvature)) {
        return NULL;
    }
    
    float* data = new float[num_particles * 6];
    for (int i = 0; i < num_particles * 6; i++) {
        PyObject* item = PyList_GetItem(data_obj, i);
        data[i] = (float)PyFloat_AsDouble(item);
    }
    
    int* results = new int[num_particles];
    
    filter_by_curvature(data, num_particles, min_curvature, max_curvature, results);
    
    PyObject* results_list = PyList_New(num_particles);
    for (int i = 0; i < num_particles; i++) {
        PyList_SetItem(results_list, i, PyLong_FromLong(results[i]));
    }
    
    delete[] data;
    delete[] results;
    
    return results_list;
}

static PyObject* py_assign_lattices(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int num_particles;
    unsigned int min_neighbours, min_lattice_size;
    
    if (!PyArg_ParseTuple(args, "OII", &data_obj, &num_particles, &min_neighbours, &min_lattice_size)) {
        return NULL;
    }
    
    float* data = new float[num_particles * 6];
    for (int i = 0; i < num_particles * 6; i++) {
        PyObject* item = PyList_GetItem(data_obj, i);
        data[i] = (float)PyFloat_AsDouble(item);
    }
    
    int* results = new int[num_particles];
    
    assign_lattices(data, num_particles, min_neighbours, min_lattice_size, results);
    
    PyObject* results_list = PyList_New(num_particles);
    for (int i = 0; i < num_particles; i++) {
        PyList_SetItem(results_list, i, PyLong_FromLong(results[i]));
    }
    
    delete[] data;
    delete[] results;
    
    return results_list;
}

static PyObject* py_clean_particles(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int num_particles;
    PyObject* params_obj;
    
    if (!PyArg_ParseTuple(args, "OiO", &data_obj, &num_particles, &params_obj)) {
        return NULL;
    }
    
    float* data = new float[num_particles * 6];
    for (int i = 0; i < num_particles * 6; i++) {
        PyObject* item = PyList_GetItem(data_obj, i);
        data[i] = (float)PyFloat_AsDouble(item);
    }
    
    // Extract parameters from Python object
    CleanParams params(
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 0)), // min_distance
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 1)), // max_distance
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 2)), // min_orientation
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 3)), // max_orientation
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 4)), // min_curvature
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 5)), // max_curvature
        (unsigned int)PyLong_AsLong(PyList_GetItem(params_obj, 6)), // min_lattice_size
        (unsigned int)PyLong_AsLong(PyList_GetItem(params_obj, 7)), // min_neighbours
        (bool)PyLong_AsLong(PyList_GetItem(params_obj, 8))  // allow_flips
    );
    
    int* results = new int[num_particles];
    
    clean_particles(data, num_particles, &params, results);
    
    PyObject* results_list = PyList_New(num_particles);
    for (int i = 0; i < num_particles; i++) {
        PyList_SetItem(results_list, i, PyLong_FromLong(results[i]));
    }
    
    delete[] data;
    delete[] results;
    
    return results_list;
}

static PyObject* py_clean_and_detect_flips(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int num_particles;
    PyObject* params_obj;
    
    if (!PyArg_ParseTuple(args, "OiO", &data_obj, &num_particles, &params_obj)) {
        return NULL;
    }
    
    float* data = new float[num_particles * 6];
    for (int i = 0; i < num_particles * 6; i++) {
        PyObject* item = PyList_GetItem(data_obj, i);
        data[i] = (float)PyFloat_AsDouble(item);
    }
    
    // Extract parameters from Python object
    CleanParams params(
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 0)), // min_distance
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 1)), // max_distance
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 2)), // min_orientation
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 3)), // max_orientation
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 4)), // min_curvature
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 5)), // max_curvature
        (unsigned int)PyLong_AsLong(PyList_GetItem(params_obj, 6)), // min_lattice_size
        (unsigned int)PyLong_AsLong(PyList_GetItem(params_obj, 7)), // min_neighbours
        (bool)PyLong_AsLong(PyList_GetItem(params_obj, 8))  // allow_flips
    );
    
    int* lattice_results = new int[num_particles];
    int* flipped_results = new int[num_particles];
    
    clean_and_detect_flips(data, num_particles, &params, lattice_results, flipped_results);
    
    // Convert results back to Python - return tuple of (lattice_results, flipped_results)
    PyObject* lattice_list = PyList_New(num_particles);
    PyObject* flipped_list = PyList_New(num_particles);
    
    for (int i = 0; i < num_particles; i++) {
        PyList_SetItem(lattice_list, i, PyLong_FromLong(lattice_results[i]));
        PyList_SetItem(flipped_list, i, PyLong_FromLong(flipped_results[i]));
    }
    
    PyObject* result_tuple = PyTuple_New(2);
    PyTuple_SetItem(result_tuple, 0, lattice_list);
    PyTuple_SetItem(result_tuple, 1, flipped_list);
    
    delete[] data;
    delete[] lattice_results;
    delete[] flipped_results;
    
    return result_tuple;
}

static PyObject* py_debug_flip_detection(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    int num_particles;
    PyObject* params_obj;
    
    if (!PyArg_ParseTuple(args, "OiO", &data_obj, &num_particles, &params_obj)) {
        return NULL;
    }
    
    float* data = new float[num_particles * 6];
    for (int i = 0; i < num_particles * 6; i++) {
        PyObject* item = PyList_GetItem(data_obj, i);
        data[i] = (float)PyFloat_AsDouble(item);
    }
    
    // Extract parameters from Python object
    CleanParams params(
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 0)), // min_distance
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 1)), // max_distance
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 2)), // min_orientation
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 3)), // max_orientation
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 4)), // min_curvature
        (float)PyFloat_AsDouble(PyList_GetItem(params_obj, 5)), // max_curvature
        (unsigned int)PyLong_AsLong(PyList_GetItem(params_obj, 6)), // min_lattice_size
        (unsigned int)PyLong_AsLong(PyList_GetItem(params_obj, 7)), // min_neighbours
        (bool)PyLong_AsLong(PyList_GetItem(params_obj, 8))  // allow_flips
    );
    
    int* lattice_results = new int[num_particles];
    int* flipped_results = new int[num_particles];
    
    debug_flip_detection(data, num_particles, &params, lattice_results, flipped_results);
    
    // Convert results back to Python - return tuple of (lattice_results, flipped_results)
    PyObject* lattice_list = PyList_New(num_particles);
    PyObject* flipped_list = PyList_New(num_particles);
    
    for (int i = 0; i < num_particles; i++) {
        PyList_SetItem(lattice_list, i, PyLong_FromLong(lattice_results[i]));
        PyList_SetItem(flipped_list, i, PyLong_FromLong(flipped_results[i]));
    }
    
    PyObject* result_tuple = PyTuple_New(2);
    PyTuple_SetItem(result_tuple, 0, lattice_list);
    PyTuple_SetItem(result_tuple, 1, flipped_list);
    
    delete[] data;
    delete[] lattice_results;
    delete[] flipped_results;
    
    return result_tuple;
}

static PyObject* py_set_log_level(PyObject* self, PyObject* args) {
    int level;
    
    if (!PyArg_ParseTuple(args, "i", &level)) {
        return NULL;
    }
    
    set_log_level(level);
    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef processing_methods[] = {
    {"find_neighbours", py_find_neighbours, METH_VARARGS, "Find neighbours based on distance"},
    {"filter_by_orientation", py_filter_by_orientation, METH_VARARGS, "Filter neighbours by orientation"},
    {"filter_by_curvature", py_filter_by_curvature, METH_VARARGS, "Filter neighbours by curvature"},
    {"assign_lattices", py_assign_lattices, METH_VARARGS, "Assign particles to lattices"},
    {"clean_particles", py_clean_particles, METH_VARARGS, "Run full particle cleaning pipeline"},
    {"clean_and_detect_flips", py_clean_and_detect_flips, METH_VARARGS, "Run cleaning and flip detection pipeline"},
    {"debug_flip_detection", py_debug_flip_detection, METH_VARARGS, "Debug flip detection with manual lattice assignment"},
    {"set_log_level", py_set_log_level, METH_VARARGS, "Set C++ log level"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Python module definition
static PyModuleDef processing_module = {
    PyModuleDef_HEAD_INIT,
    "processing_cpp",
    "C++ processing functions for MagpiEM",
    -1,
    processing_methods
};

// Python module initialization function
PyMODINIT_FUNC PyInit_processing_cpp(void) {
    return PyModule_Create(&processing_module);
}
