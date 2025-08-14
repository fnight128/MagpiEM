#define BUILDING_DLL
#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

static std::vector<Particle> g_particles;
static bool g_particles_initialized = false;

EXPORT void clean_particles(float* data, int num_particles, float min_distance_squared, float max_distance_squared, int* results) {
    printf("Processing %d particles with neighbor range: %.2f - %.2f units\n", 
           num_particles, min_distance_squared, max_distance_squared);
    g_particles = Particle::from_raw_data(data, num_particles);
    g_particles_initialized = true;
    
    // Calculate neighbors for all particles
    for (int i = 0; i < num_particles; i++) {
        Particle& current = g_particles[i];
        current.neighbors.clear();
        
        for (int j = 0; j < num_particles; j++) {
            if (i == j) continue;
            
            Particle& other = g_particles[j];
            float distance_squared = current.calculate_distance_squared(other);

            if (distance_squared >= min_distance_squared && distance_squared <= max_distance_squared) {
                current.neighbors.push_back(&other);
            }
        }
        results[i] = current.get_neighbor_count();
    }
}



// Function to reset the global state (useful for testing)
void reset_particles() {
    g_particles.clear();
    g_particles_initialized = false;
}
