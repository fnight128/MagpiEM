#define BUILDING_DLL
#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

static std::vector<Particle> g_particles;
static bool g_particles_initialized = false;

EXPORT void find_neighbours(float* data, int num_particles, float min_distance_squared, float max_distance_squared, int* results) {
    printf("Finding neighbours for %d particles (distance: %.2f - %.2f)\n", num_particles, min_distance_squared, max_distance_squared);
    
    g_particles = Particle::from_raw_data(data, num_particles);
    g_particles_initialized = true;
    
    // Calculate neighbours for all particles based on distance only
    for (int i = 0; i < num_particles; i++) {
        Particle& current = g_particles[i];
        current.neighbours.clear();
        current.neighbours.reserve(num_particles / 10); // Reserve some space to avoid reallocations
        
        for (int j = 0; j < num_particles; j++) {
            if (i == j) continue;
            
            Particle& other = g_particles[j];
            float distance_squared = current.calculate_distance_squared(other);
            
            // Check distance criteria using squared distances
            if (distance_squared >= min_distance_squared && distance_squared <= max_distance_squared) {
                current.neighbours.push_back(&other);
            }
        }
        
        results[i] = current.get_neighbour_count();
    }
}

// Generic filter function using a predicate
template<typename Predicate>
void filter_neighbours_generic(float* data, int num_particles, const std::string& filter_name, 
                              float min_value, float max_value, int* results, Predicate predicate) {
    printf("Filtering neighbours by %s for %d particles (%s: %.2f - %.2f)\n", 
           filter_name.c_str(), num_particles, filter_name.c_str(), min_value, max_value);
    
    // Reuse existing particles if already initialized, otherwise create new ones
    if (!g_particles_initialized) {
        g_particles = Particle::from_raw_data(data, num_particles);
        g_particles_initialized = true;
    }
    
    // Filter existing neighbours using the provided predicate
    for (int i = 0; i < num_particles; i++) {
        Particle& current = g_particles[i];
        std::vector<Particle*> valid_neighbours;
        valid_neighbours.reserve(current.neighbours.size());
        
        for (Particle* neighbour : current.neighbours) {
            float value = predicate(current, *neighbour);
            
            // Check if value is within range
            if (value >= min_value && value <= max_value) {
                valid_neighbours.push_back(neighbour);
            }
        }
        
        current.neighbours = std::move(valid_neighbours);
        results[i] = current.get_neighbour_count();
    }
}

EXPORT void filter_by_orientation(float* data, int num_particles, float min_orientation, float max_orientation, int* results) {
    auto orientation_predicate = [](const Particle& current, const Particle& neighbour) -> float {
        float dot_product = Particle::dot_product(current.orientation, neighbour.orientation);
        return std::max(-1.0f, std::min(1.0f, dot_product));
    };
    
    filter_neighbours_generic(data, num_particles, "orientation", min_orientation, max_orientation, results, orientation_predicate);
}

EXPORT void filter_by_curvature(float* data, int num_particles, float min_curvature, float max_curvature, int* results) {
    auto curvature_predicate = [](const Particle& current, const Particle& neighbour) -> float {
        return current.curvature(neighbour);
    };
    
    filter_neighbours_generic(data, num_particles, "curvature", min_curvature, max_curvature, results, curvature_predicate);
}

EXPORT void clean_particles(float* data, int num_particles, CleanParams* params, int* results) {
    printf("Processing %d particles with parameters:\n", num_particles);
    printf("  Distance: %.2f - %.2f\n", params->min_distance, params->max_distance);
    printf("  Orientation: %.2f - %.2f\n", params->min_orientation, params->max_orientation);
    printf("  Curvature: %.2f - %.2f\n", params->min_curvature, params->max_curvature);
    printf("  Min lattice size: %d, Min neighbours: %d\n", params->min_lattice_size, params->min_neighbours);
    
    // First find neighbours by distance
    find_neighbours(data, num_particles, params->min_distance, params->max_distance, results);
    
    // Then filter by orientation
    filter_by_orientation(data, num_particles, params->min_orientation, params->max_orientation, results);
    
    // Finally filter by curvature
    filter_by_curvature(data, num_particles, params->min_curvature, params->max_curvature, results);
}

// Function to reset the global state (useful for testing)
void reset_particles() {
    g_particles.clear();
    g_particles_initialized = false;
}
