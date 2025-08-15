#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

static std::vector<Particle> g_particles;
static bool g_particles_initialized = false;

EXPORT void clean_particles(float* data, int num_particles, CleanParams* params, int* results) {
    printf("Processing %d particles with parameters:\n", num_particles);
    printf("  Distance: %.2f - %.2f\n", params->min_distance, params->max_distance);
    printf("  Orientation: %.2f - %.2f\n", params->min_orientation, params->max_orientation);
    printf("  Curvature: %.2f - %.2f\n", params->min_curvature, params->max_curvature);
    printf("  Min lattice size: %d, Min neighbors: %d\n", params->min_lattice_size, params->min_neighbors);
    
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
            
            // Check distance criteria
            if (distance_squared < params->min_distance || 
                distance_squared > params->max_distance) {
                continue;
            }
            
            // TODO: Add orientation and curvature checks here
            // For now, just use distance criteria
            
            current.neighbors.push_back(&other);
        }
        
        results[i] = current.get_neighbor_count();
    }
}

// Function to reset the global state (useful for testing)
void reset_particles() {
    g_particles.clear();
    g_particles_initialized = false;
}
