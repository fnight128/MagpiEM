#define BUILDING_DLL
#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <map>

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
    printf("Assigning lattices for %d particles (min neighbours: %d, min lattice size: %d)\n", num_particles, min_neighbours, min_lattice_size);
    
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
    std::vector<std::vector<Particle*>> lattice_groups;
    
    // Assign lattices
    for (int i = 0; i < num_particles; i++) {
        Particle& current = g_particles[i];
        
        // Skip if already assigned or insufficient neighbours
        if (current.lattice != 0 || current.get_neighbour_count() < min_neighbours) {
            continue;
        }
        
        // Start a new lattice with this particle
        current.lattice = next_lattice_id;
        
        // Use breadth-first search to assign all connected particles to the same lattice
        std::vector<Particle*> queue;
        std::vector<Particle*> lattice_members;
        queue.push_back(&current);
        lattice_members.push_back(&current);
        
        for (size_t queue_idx = 0; queue_idx < queue.size(); queue_idx++) {
            Particle* particle = queue[queue_idx];
            
            // Add all unassigned neighbours with sufficient neighbours to the queue
            for (Particle* neighbour : particle->neighbours) {
                if (neighbour->lattice == 0 && neighbour->get_neighbour_count() >= min_neighbours) {
                    neighbour->lattice = next_lattice_id;
                    queue.push_back(neighbour);
                    lattice_members.push_back(neighbour);
                }
            }
        }
        
        // Store the lattice group for later size checking
        lattice_groups.push_back(lattice_members);
        next_lattice_id++;
    }
    
    // Filter out lattices that are too small
    printf("Checking %zu lattices for minimum size %d\n", lattice_groups.size(), min_lattice_size);
    for (size_t i = 0; i < lattice_groups.size(); i++) {
        const auto& lattice_group = lattice_groups[i];
        printf("  Lattice %zu: %zu particles", i + 1, lattice_group.size());
        if (lattice_group.size() < min_lattice_size) {
            printf(" -> REMOVED (too small)");
            // Reassign all particles in this lattice to 0
            for (Particle* particle : lattice_group) {
                particle->lattice = 0;
            }
        } else {
            printf(" -> KEPT");
        }
        printf("\n");
    }
    
    // Copy lattice assignments to results array
    for (int i = 0; i < num_particles; i++) {
        results[i] = g_particles[i].lattice;
    }
    
    // Print summary of final lattice assignments
    std::map<int, int> lattice_counts;
    for (int i = 0; i < num_particles; i++) {
        lattice_counts[g_particles[i].lattice]++;
    }
    printf("Final lattice assignments:\n");
    for (const auto& [lattice_id, count] : lattice_counts) {
        if (lattice_id == 0) {
            printf("  Unassigned (lattice 0): %d particles\n", count);
        } else {
            printf("  Lattice %d: %d particles\n", lattice_id, count);
        }
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
