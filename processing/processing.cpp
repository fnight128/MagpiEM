#define BUILDING_DLL
#include "processing.hpp"

#include <iostream>
using namespace std;

EXPORT void clean_particles(float* data, int num_points, int* results) {
    printf("Processing %d oriented points\n", num_points);
    
    Particle* points = reinterpret_cast<Particle*>(data);
    
    for (int i = 0; i < num_points; i++) {
        Particle& point = points[i];
        
        printf("      Point %d: pos(%.2f,%.2f,%.2f) orient(%.2f,%.2f,%.2f)\n",
               i, point.x, point.y, point.z, point.rx, point.ry, point.rz);

        results[i] = static_cast<int>(point.x + point.y + point.z);
    }
}
