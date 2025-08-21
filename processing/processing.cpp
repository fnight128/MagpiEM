#define BUILDING_DLL
#include "processing.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <cstring>
#include <cstdarg>

enum LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3
};

static LogLevel g_log_level = LOG_WARNING;

void log_message(LogLevel level, const char* format, ...) {
    if (level < g_log_level) {
        return; 
    }
    
    const char* level_names[] = {"DEBUG", "INFO", "WARNING", "ERROR"};
    
    va_list args;
    va_start(args, format);

    va_list args_copy;
    va_copy(args_copy, args);
    int len = vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);
    
    if (len > 0) {
        std::string message(len + 1, '\0');
        vsnprintf(&message[0], len + 1, format, args);
        message.resize(len);
        
        std::cerr << "C++ [" << level_names[level] << "] " << message << std::endl;
    }
    
    va_end(args);
}

EXPORT void set_log_level(int level) {
    g_log_level = static_cast<LogLevel>(level);
}

// Particle processor class to manage state
class ParticleProcessor {
private:
    std::vector<Particle> particles;
    bool initialized = false;
    std::vector<Particle*> temp_neighbours; // Reusable buffer for filtering

public:
    void reset() {
        particles.clear();
        initialized = false;
    }

    void ensure_initialized(float* data, int num_particles) {
        if (!initialized) {
            particles = Particle::from_raw_data(data, num_particles);
            initialized = true;
        }
    }

    void find_neighbours(float* data, int num_particles, float min_distance_squared, float max_distance_squared, int* results) {
        log_message(LOG_INFO, "Finding neighbours for %d particles (distance: %.2f - %.2f)", num_particles, min_distance_squared, max_distance_squared);
        
        ensure_initialized(data, num_particles);
        
        // Clear all neighbour lists
        for (auto& particle : particles) {
            particle.neighbours.clear();
            particle.neighbours.reserve(num_particles / 10); // Reserve some space to avoid reallocations
        }
        
        // Calculate neighbours for all particles based on distance only
        // Only compare each pair once to avoid double calculations
        for (int i = 0; i < num_particles; i++) {
            Particle& current = particles[i];
            
            for (int j = i + 1; j < num_particles; j++) {
                Particle& other = particles[j];
                float distance_squared = current.calculate_distance_squared(other);
                
                // Check distance criteria using squared distances
                if (distance_squared >= min_distance_squared && distance_squared <= max_distance_squared) {
                    // Add bidirectional neighbours
                    current.neighbours.push_back(&other);
                    other.neighbours.push_back(&current);
                }
            }
            
            results[i] = current.get_neighbour_count();
        }
    }

    // Generic filter function using a predicate with vectorised optimisations
    template<typename Predicate>
    void filter_neighbours_generic(float* data, int num_particles, const std::string& filter_name, 
                                  float min_value, float max_value, int* results, Predicate predicate) {
        log_message(LOG_INFO, "Filtering neighbours by %s for %d particles (%s: %.2f - %.2f)", 
                   filter_name.c_str(), num_particles, filter_name.c_str(), min_value, max_value);
        
        ensure_initialized(data, num_particles);
        
        // Filter existing neighbours using the provided predicate
        for (int i = 0; i < num_particles; i++) {
            Particle& current = particles[i];
            temp_neighbours.clear();
            temp_neighbours.reserve(current.neighbours.size());
            
            for (Particle* neighbour : current.neighbours) {
                float value = predicate(current, *neighbour);
                
                // Check if value is within range
                if (value >= min_value && value <= max_value) {
                    temp_neighbours.push_back(neighbour);
                }
            }
            
            current.neighbours = std::move(temp_neighbours);
            results[i] = current.get_neighbour_count();
        }
    }

    void filter_by_orientation(float* data, int num_particles, float min_orientation, float max_orientation, int* results) {
        auto orientation_predicate = [](const Particle& current, const Particle& neighbour) -> float {
            float dot_product = Particle::dot_product(current.orientation, neighbour.orientation);
            return std::max(-1.0f, std::min(1.0f, dot_product));
        };
        
        filter_neighbours_generic(data, num_particles, "orientation", min_orientation, max_orientation, results, orientation_predicate);
    }

    void filter_by_curvature(float* data, int num_particles, float min_curvature, float max_curvature, int* results) {
        auto curvature_predicate = [](const Particle& current, const Particle& neighbour) -> float {
            return current.curvature(neighbour);
        };
        
        filter_neighbours_generic(data, num_particles, "curvature", min_curvature, max_curvature, results, curvature_predicate);
    }

    void assign_lattices(float* data, int num_particles, unsigned int min_neighbours, unsigned int min_lattice_size, int* results) {
        ensure_initialized(data, num_particles);
        
        // Reset all lattice assignments
        for (auto& particle : particles) {
            particle.lattice = 0;
        }
        
        unsigned int next_lattice_id = 1;
        
        // Assign lattices
        for (int i = 0; i < num_particles; i++) {
            Particle& current = particles[i];
            
            // Skip if already assigned or insufficient neighbours
            if (current.lattice != 0 || current.get_neighbour_count() < min_neighbours) {
                continue;
            }
            
            // Start a new lattice with this particle
            current.lattice = next_lattice_id;
            
            // Use optimized breadth-first search with pre-allocated queue
            std::vector<Particle*> queue;
            queue.reserve(num_particles);
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
        std::vector<unsigned int> lattice_sizes(next_lattice_id, 0);
        for (const auto& particle : particles) {
            if (particle.lattice > 0) {
                lattice_sizes[particle.lattice]++;
            }
        }
        
        // Reassign particles in small lattices to 0
        for (auto& particle : particles) {
            if (particle.lattice > 0 && lattice_sizes[particle.lattice] < min_lattice_size) {
                particle.lattice = 0;
            }
        }
        
        // Copy lattice assignments to results array
        for (int i = 0; i < num_particles; i++) {
            results[i] = particles[i].lattice;
        }
    }

    void clean_particles(float* data, int num_particles, CleanParams* params, int* results) {
        log_message(LOG_INFO, "Processing %d particles with parameters:", num_particles);
        log_message(LOG_DEBUG, "  Distance: %.2f - %.2f", params->min_distance, params->max_distance);
        log_message(LOG_DEBUG, "  Orientation: %.2f - %.2f", params->min_orientation, params->max_orientation);
        log_message(LOG_DEBUG, "  Curvature: %.2f - %.2f", params->min_curvature, params->max_curvature);
        log_message(LOG_DEBUG, "  Min lattice size: %d, Min neighbours: %d", params->min_lattice_size, params->min_neighbours);
        
        // Reset state to ensure we process the new data and avoid leakage
        reset();
        
        find_neighbours(data, num_particles, params->min_distance, params->max_distance, results);
        filter_by_orientation(data, num_particles, params->min_orientation, params->max_orientation, results);
        filter_by_curvature(data, num_particles, params->min_curvature, params->max_curvature, results);
        assign_lattices(data, num_particles, params->min_neighbours, params->min_lattice_size, results);
    }

    void get_cleaned_neighbours(float* data, int num_particles, CleanParams* params, int* offsets, int* neighbours_out) {
        // Reset and run pipeline up to curvature
        reset();

        // Distance neighbours
        find_neighbours(data, num_particles, params->min_distance, params->max_distance, offsets /*reuse temp buffer*/);
        // Orientation filter
        filter_by_orientation(data, num_particles, params->min_orientation, params->max_orientation, offsets);
        // Curvature filter
        filter_by_curvature(data, num_particles, params->min_curvature, params->max_curvature, offsets);

        // Build CSR outputs from particles.neighbours
        int running_total = 0;
        for (int i = 0; i < num_particles; ++i) {
            offsets[i] = running_total;
            running_total += static_cast<int>(particles[i].neighbours.size());
        }
        offsets[num_particles] = running_total;

        if (neighbours_out == nullptr) {
            return;
        }
        // Fill flattened neighbour indices by particle index position in particles
        int cursor = 0;
        for (int i = 0; i < num_particles; ++i) {
            for (Particle* neighbour : particles[i].neighbours) {
                // neighbour pointer points into particles; compute index by pointer arithmetic
                int neighbour_index = static_cast<int>(neighbour - &particles[0]);
                neighbours_out[cursor++] = neighbour_index;
            }
        }
    }
};

// Global instance for the particle processor
static ParticleProcessor g_processor;

// Direct exports of the processor methods
EXPORT void find_neighbours(float* data, int num_particles, float min_distance_squared, float max_distance_squared, int* results) {
    g_processor.find_neighbours(data, num_particles, min_distance_squared, max_distance_squared, results);
}

EXPORT void filter_by_orientation(float* data, int num_particles, float min_orientation, float max_orientation, int* results) {
    g_processor.filter_by_orientation(data, num_particles, min_orientation, max_orientation, results);
}

EXPORT void filter_by_curvature(float* data, int num_particles, float min_curvature, float max_curvature, int* results) {
    g_processor.filter_by_curvature(data, num_particles, min_curvature, max_curvature, results);
}

EXPORT void assign_lattices(float* data, int num_particles, unsigned int min_neighbours, unsigned int min_lattice_size, int* results) {
    g_processor.assign_lattices(data, num_particles, min_neighbours, min_lattice_size, results);
}

EXPORT void clean_particles(float* data, int num_particles, CleanParams* params, int* results) {
    g_processor.clean_particles(data, num_particles, params, results);
}

EXPORT void get_cleaned_neighbours(float* data, int num_particles, CleanParams* params, int* offsets, int* neighbours_out) {
    g_processor.get_cleaned_neighbours(data, num_particles, params, offsets, neighbours_out);
}

// Function to reset the processor state (useful for testing)
void reset_particles() {
    g_processor.reset();
}
