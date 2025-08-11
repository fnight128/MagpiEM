#define BUILDING_DLL
#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

float calculate_distance(const Particle& p1, const Particle& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return dx*dx + dy*dy + dz*dz;
}

EXPORT void clean_particles(float* data, int num_particles, float min_distance_squared, float max_distance_squared, int* results) {
    printf("Processing %d particles with neighbor range: %.2f - %.2f units\n", 
           num_particles, min_distance_squared, max_distance_squared);
    
    Particle* particles = reinterpret_cast<Particle*>(data);
    
    for (int i = 0; i < num_particles; i++) {
        Particle& current = particles[i];
        int neighbor_count = 0;
        
        printf("  Particle %d: pos(%.2f,%.2f,%.2f) orient(%.2f,%.2f,%.2f)\n",
               i, current.x, current.y, current.z, current.rx, current.ry, current.rz);
        
        for (int j = 0; j < num_particles; j++) {
            if (i == j) continue;
            
            Particle& other = particles[j];
            float distance = calculate_distance(current, other);

            if (distance >= min_distance_squared && distance <= max_distance_squared) {
                neighbor_count++;
                printf("    -> Neighbor %d at distance %.2f\n", j, distance);
            }
        }
        
        // Store the neighbor count as the result for this particle
        results[i] = neighbor_count;
        printf("    Total neighbors: %d\n", neighbor_count);
    }
}
