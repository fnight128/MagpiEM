#define BUILDING_DLL
#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <cstring>

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

EXPORT void assign_lattices(float* data, int num_particles, int min_neighbours, int min_lattice_size, int* results) {
    // Reuse existing particles if already initialized, otherwise create new ones
    if (!g_particles_initialized) {
        g_particles = Particle::from_raw_data(data, num_particles);
        g_particles_initialized = true;
    }
    
    // Reset all lattice assignments
    for (auto& particle : g_particles) {
        particle.lattice = 0;
    }
    
    int next_lattice_id = 1;
    
    // Assign lattices using optimized BFS
    for (int i = 0; i < num_particles; i++) {
        Particle& current = g_particles[i];
        
        // Skip if already assigned or insufficient neighbours
        if (current.lattice != 0 || current.get_neighbour_count() < min_neighbours) {
            continue;
        }
        
        // Start a new lattice with this particle
        current.lattice = next_lattice_id;
        
        // Use optimized breadth-first search with pre-allocated queue
        std::vector<Particle*> queue;
        queue.reserve(num_particles); // Pre-allocate to avoid reallocations
        queue.push_back(&current);
        
        size_t queue_idx = 0;
        while (queue_idx < queue.size()) {
            Particle* particle = queue[queue_idx++];
            
            // Add all unassigned neighbours with sufficient neighbours to the queue
            for (Particle* neighbour : particle->neighbours) {
                if (neighbour->lattice == 0 && neighbour->get_neighbour_count() >= min_neighbours) {
                    neighbour->lattice = next_lattice_id;
                    queue.push_back(neighbour);
                }
            }
        }
        
        next_lattice_id++;
    }
    
    // Filter out small lattices in a single pass
    std::vector<int> lattice_sizes(next_lattice_id, 0);
    for (const auto& particle : g_particles) {
        if (particle.lattice > 0) {
            lattice_sizes[particle.lattice]++;
        }
    }
    
    // Reassign particles in small lattices to 0
    for (auto& particle : g_particles) {
        if (particle.lattice > 0 && lattice_sizes[particle.lattice] < min_lattice_size) {
            particle.lattice = 0;
        }
    }
    
    // Copy lattice assignments to results array
    for (int i = 0; i < num_particles; i++) {
        results[i] = g_particles[i].lattice;
    }
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
    
    // Then filter by curvature
    filter_by_curvature(data, num_particles, params->min_curvature, params->max_curvature, results);
    
    // Finally assign lattices
    assign_lattices(data, num_particles, params->min_neighbours, params->min_lattice_size, results);
}

// Function to reset the global state (useful for testing)
void reset_particles() {
    g_particles.clear();
    g_particles_initialized = false;
}

// Debug/testing utility to extract neighbour IDs after applying distance, orientation, curvature
// The function fills offsets (size: num_particles + 1) and neighbours_out (flattened IDs)
// If neighbours_out is nullptr, only computes offsets and total length in offsets[num_particles]
EXPORT void get_cleaned_neighbours(float* data, int num_particles, CleanParams* params, int* offsets, int* neighbours_out) {
    // Reset and run pipeline up to curvature
    reset_particles();

    // Distance neighbours
    find_neighbours(data, num_particles, params->min_distance, params->max_distance, offsets /*reuse temp buffer*/);
    // Orientation filter
    filter_by_orientation(data, num_particles, params->min_orientation, params->max_orientation, offsets);
    // Curvature filter
    filter_by_curvature(data, num_particles, params->min_curvature, params->max_curvature, offsets);

    // Build CSR outputs from g_particles.neighbours
    int running_total = 0;
    for (int i = 0; i < num_particles; ++i) {
        offsets[i] = running_total;
        running_total += static_cast<int>(g_particles[i].neighbours.size());
    }
    offsets[num_particles] = running_total;

    if (neighbours_out == nullptr) {
        return;
    }
    // Fill flattened neighbour indices by particle index position in g_particles
    int cursor = 0;
    for (int i = 0; i < num_particles; ++i) {
        for (Particle* neighbour : g_particles[i].neighbours) {
            // neighbour pointer points into g_particles; compute index by pointer arithmetic
            int neighbour_index = static_cast<int>(neighbour - &g_particles[0]);
            neighbours_out[cursor++] = neighbour_index;
        }
    }
}
