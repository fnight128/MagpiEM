#define BUILDING_DLL
#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

float calculate_distance_squared(const Particle& p1, const Particle& p2) {
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
        
        for (int j = 0; j < num_particles; j++) {
            if (i == j) continue;
            
            Particle& other = particles[j];
            float distance_squared = calculate_distance_squared(current, other);

            if (distance_squared >= min_distance_squared && distance_squared <= max_distance_squared) {
                neighbor_count++;
            }
        }
        results[i] = neighbor_count;
    }
}
