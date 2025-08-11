#ifndef PROCESSING_HPP
#define PROCESSING_HPP

#ifdef _WIN32
    #ifdef BUILDING_DLL
        #define EXPORT __declspec(dllexport)
    #else
        #define EXPORT __declspec(dllimport)
    #endif
#else
    #define EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct Particle {
    float x, y, z;
    float rx, ry, rz;
};

EXPORT void clean_particles(float* data, int num_points, float min_distance, float max_distance, int* results);

#ifdef __cplusplus
}
#endif

#endif //PROCESSING_HPP
