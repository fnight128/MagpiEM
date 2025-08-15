#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

static std::vector<Particle> g_particles;
static bool g_particles_initialized = false;

EXPORT void find_neighbors(float* data, int num_particles, float min_distance, float max_distance, int* results) {
    printf("Finding neighbors for %d particles (distance: %.2f - %.2f)\n", num_particles, min_distance, max_distance);
    
    g_particles = Particle::from_raw_data(data, num_particles);
    g_particles_initialized = true;
    
    // Calculate neighbors for all particles based on distance only
    for (int i = 0; i < num_particles; i++) {
        Particle& current = g_particles[i];
        current.neighbors.clear();
        
        for (int j = 0; j < num_particles; j++) {
            if (i == j) continue;
            
            Particle& other = g_particles[j];
            float distance_squared = current.calculate_distance_squared(other);
            
            // Check distance criteria
            if (distance_squared >= min_distance && distance_squared <= max_distance) {
                current.neighbors.push_back(&other);
            }
        }
        
        results[i] = current.get_neighbor_count();
    }
}

EXPORT void filter_by_orientation(float* data, int num_particles, float min_orientation, float max_orientation, int* results) {
    printf("Filtering neighbors by orientation for %d particles (orientation: %.2f - %.2f)\n", num_particles, min_orientation, max_orientation);
    
    // Reuse existing particles if already initialized, otherwise create new ones
    if (!g_particles_initialized) {
        g_particles = Particle::from_raw_data(data, num_particles);
        g_particles_initialized = true;
    }
    
    // Normalize all orientation vectors
    for (int i = 0; i < num_particles; i++) {
        Particle::normalize_vector(g_particles[i].rx, g_particles[i].ry, g_particles[i].rz);
    }
    
    // Filter existing neighbors by orientation
    for (int i = 0; i < num_particles; i++) {
        Particle& current = g_particles[i];
        
        auto it = current.neighbors.begin();
        while (it != current.neighbors.end()) {
            Particle* neighbor = *it;
            
            // Calculate dot product of orientations (clamped between -1 and 1)
            float dot_product = Particle::dot_product(
                current.rx, current.ry, current.rz,
                neighbor->rx, neighbor->ry, neighbor->rz
            );
            dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
            
            // Check if orientation angle is within range
            if (dot_product < min_orientation || dot_product > max_orientation) {
                it = current.neighbors.erase(it);
            } else {
                ++it;
            }
        }
        
        results[i] = current.get_neighbor_count();
    }
}

EXPORT void clean_particles(float* data, int num_particles, CleanParams* params, int* results) {
    printf("Processing %d particles with parameters:\n", num_particles);
    printf("  Distance: %.2f - %.2f\n", params->min_distance, params->max_distance);
    printf("  Orientation: %.2f - %.2f\n", params->min_orientation, params->max_orientation);
    printf("  Curvature: %.2f - %.2f\n", params->min_curvature, params->max_curvature);
    printf("  Min lattice size: %d, Min neighbors: %d\n", params->min_lattice_size, params->min_neighbors);
    
    // First find neighbors by distance
    find_neighbors(data, num_particles, params->min_distance, params->max_distance, results);
    
    // Then filter by orientation
    filter_by_orientation(data, num_particles, params->min_orientation, params->max_orientation, results);
}

// Function to reset the global state (useful for testing)
void reset_particles() {
    g_particles.clear();
    g_particles_initialized = false;
}
