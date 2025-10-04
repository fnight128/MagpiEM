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

// Forward declaration for template function
template<typename ParticlePtrType>
void perform_flip_detection(const std::vector<Particle>& particles, const std::map<unsigned int, std::vector<ParticlePtrType>>& lattice_groups, 
                           const CleanParams* params, int* flipped_results);

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

void set_log_level(int level) {
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

    // Generic filter function using a predicate that returns bool
    template<typename Predicate>
    void filter_neighbours_generic(float* data, int num_particles, const std::string& filter_name, 
                                  int* results, Predicate predicate) {
        ensure_initialized(data, num_particles);
        
        for (int i = 0; i < num_particles; i++) {
            Particle& current = particles[i];
            temp_neighbours.clear();
            temp_neighbours.reserve(current.neighbours.size());
            
            for (Particle* neighbour : current.neighbours) {
                if (predicate(current, *neighbour)) {
                    temp_neighbours.push_back(neighbour);
                }
            }
            
            current.neighbours = std::move(temp_neighbours);
            results[i] = current.get_neighbour_count();
        }
    }

    void filter_by_orientation(float* data, int num_particles, float min_orientation, float max_orientation, bool allow_flips, int* results) {
        log_message(LOG_INFO, "Filtering neighbours by orientation for %d particles (orientation: %.2f - %.2f%s)", 
                   num_particles, min_orientation, max_orientation,
                   allow_flips ? ", allow_flips: true" : "");
        
        if (allow_flips) {
            auto orientation_predicate_flips = [min_orientation, max_orientation](const Particle& current, const Particle& neighbour) -> bool {
                float dot_product = Particle::dot_product(current.orientation, neighbour.orientation);
                dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
                return (dot_product >= min_orientation && dot_product <= max_orientation) ||
                       (dot_product >= -max_orientation && dot_product <= -min_orientation);
            };
            filter_neighbours_generic(data, num_particles, "orientation", results, orientation_predicate_flips);
        } else {
            auto orientation_predicate = [min_orientation, max_orientation](const Particle& current, const Particle& neighbour) -> bool {
                float dot_product = Particle::dot_product(current.orientation, neighbour.orientation);
                dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
                return dot_product >= min_orientation && dot_product <= max_orientation;
            };
            filter_neighbours_generic(data, num_particles, "orientation", results, orientation_predicate);
        }
    }

    void filter_by_curvature(float* data, int num_particles, float min_curvature, float max_curvature, int* results) {
        log_message(LOG_INFO, "Filtering neighbours by curvature for %d particles (curvature: %.2f - %.2f)", 
                   num_particles, min_curvature, max_curvature);
        
        auto curvature_predicate = [min_curvature, max_curvature](const Particle& current, const Particle& neighbour) -> bool {
            float curvature_value = current.curvature(neighbour);
            return curvature_value >= min_curvature && curvature_value <= max_curvature;
        };
        
        filter_neighbours_generic(data, num_particles, "curvature", results, curvature_predicate);
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
        log_message(LOG_DEBUG, "  Allow flips: %s", params->allow_flips ? "true" : "false");
        
        // Reset state to ensure we process the new data and avoid leakage
        reset();
        
        find_neighbours(data, num_particles, params->min_distance, params->max_distance, results);
        filter_by_orientation(data, num_particles, params->min_orientation, params->max_orientation, params->allow_flips, results);
        filter_by_curvature(data, num_particles, params->min_curvature, params->max_curvature, results);
        assign_lattices(data, num_particles, params->min_neighbours, params->min_lattice_size, results);
    }

    void get_cleaned_neighbours(float* data, int num_particles, CleanParams* params, int* offsets, int* neighbours_out) {
        // Reset and run pipeline up to curvature
        reset();

        // Distance neighbours
        find_neighbours(data, num_particles, params->min_distance, params->max_distance, offsets /*reuse temp buffer*/);
        // Orientation filter
        filter_by_orientation(data, num_particles, params->min_orientation, params->max_orientation, params->allow_flips, offsets);
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

    // Get access to the processed particles (for flip detection)
    const std::vector<Particle>& get_particles() const {
        return particles;
    }
};

// Global instance for the particle processor
static ParticleProcessor g_processor;

// Direct exports of the processor methods
void find_neighbours(float* data, int num_particles, float min_distance_squared, float max_distance_squared, int* results) {
    g_processor.find_neighbours(data, num_particles, min_distance_squared, max_distance_squared, results);
}

void filter_by_orientation(float* data, int num_particles, float min_orientation, float max_orientation, bool allow_flips, int* results) {
    g_processor.filter_by_orientation(data, num_particles, min_orientation, max_orientation, allow_flips, results);
}

void filter_by_curvature(float* data, int num_particles, float min_curvature, float max_curvature, int* results) {
    g_processor.filter_by_curvature(data, num_particles, min_curvature, max_curvature, results);
}

void assign_lattices(float* data, int num_particles, unsigned int min_neighbours, unsigned int min_lattice_size, int* results) {
    g_processor.assign_lattices(data, num_particles, min_neighbours, min_lattice_size, results);
}

void clean_particles(float* data, int num_particles, CleanParams* params, int* results) {
    g_processor.clean_particles(data, num_particles, params, results);
}

void clean_and_detect_flips(float* data, int num_particles, CleanParams* params, int* lattice_results, int* flipped_results) {
    log_message(LOG_INFO, "Running combined cleaning and flip detection on %d particles", num_particles);
    
    // First run the normal cleaning pipeline - this is the key part!
    g_processor.clean_particles(data, num_particles, params, lattice_results);
    
    // Initialize all particles as not flipped
    for (int i = 0; i < num_particles; i++) {
        flipped_results[i] = 0;
    }
    
    // Only run flip detection if allow_flips is enabled
    if (!params->allow_flips) {
        log_message(LOG_WARNING, "Flip detection was called, but flips were not allowed in cleaning params. This is likely incorrect.");
        return;
    }
    
    // processed particles already have assigned neighbours and lattices 
    const std::vector<Particle>& particles = get_processed_particles();
    
    std::map<unsigned int, std::vector<const Particle*>> lattice_groups;
    for (const auto& particle : particles) {
        if (particle.lattice > 0) {
            lattice_groups[particle.lattice].push_back(&particle);
        }
    }
    
    log_message(LOG_DEBUG, "Processing %zu lattice groups for flip detection", lattice_groups.size());
    perform_flip_detection(particles, lattice_groups, params, flipped_results);
    
    log_message(LOG_INFO, "Flip detection completed");
}

// Debug function for testing flip detection with manual lattice assignment
void debug_flip_detection(float* data, int num_particles, CleanParams* params, int* lattice_results, int* flipped_results) noexcept(false) {
    log_message(LOG_INFO, "Running debug flip detection on %d particles", num_particles);
    
    // Initialize all particles as not flipped and assign to lattice 1
    for (int i = 0; i < num_particles; i++) {
        flipped_results[i] = 0;
        lattice_results[i] = 1; // All particles go to lattice 1
    }
    
    // Only run flip detection if allow_flips is enabled
    if (!params->allow_flips) {
        throw std::invalid_argument( "Attempted to test flip detection with flip detection disabled in cleaning params (allow_flips=False).");
        return;
    }
    
    // Find neighbours. No additional cleaning to ensure all particles included in debugging
    g_processor.find_neighbours(data, num_particles, params->min_distance, params->max_distance, lattice_results);
    
    // Get the particles with their neighbours already set
    const std::vector<Particle>& particles = get_processed_particles();
    
    // Group all particles into lattice 1 for testing
    std::map<unsigned int, std::vector<const Particle*>> lattice_groups;
    for (const auto& particle : particles) {
        lattice_groups[1].push_back(&particle);
    }
    
    perform_flip_detection(particles, lattice_groups, params, flipped_results);
    
    log_message(LOG_INFO, "Debug flip detection completed");
}

template<typename ParticlePtrType>
void perform_flip_detection(const std::vector<Particle>& particles, const std::map<unsigned int, std::vector<ParticlePtrType>>& lattice_groups, 
                           const CleanParams* params, int* flipped_results) {
    for (auto& [lattice_id, lattice_particles] : lattice_groups) {
        if (lattice_particles.size() < 3) {
            continue; // Need at least 3 particles
        }
        
        // Use a map to store direction assignments (1 or -1)
        std::map<ParticlePtrType, int> directions;
        
        // Start with first particle and assign arbitrarydirection
        ParticlePtrType random_particle = lattice_particles[0];
        directions[random_particle] = 1;
        
        // Recursively assign directions using BFS
        std::vector<ParticlePtrType> to_process = {random_particle};
        while (!to_process.empty()) {
            ParticlePtrType current = to_process.back();
            to_process.pop_back();
            
            for (ParticlePtrType neighbour : current->neighbours) {
                if (directions.find(neighbour) == directions.end()) {
                    float dot_prod = Particle::dot_product(current->orientation, neighbour->orientation);
                    if (dot_prod >= params->min_orientation && dot_prod <= params->max_orientation) {
                        // Aligned orientation: same direction
                        directions[neighbour] = directions[current];
                    } else {
                        // Not aligned: opposite direction
                        directions[neighbour] = -directions[current];
                    }
                    to_process.push_back(neighbour);
                }
            }
        }
        
        // Count parallel vs antiparallel particles
        int parallel_count = 0;
        int antiparallel_count = 0;
        for (ParticlePtrType particle : lattice_particles) {
            if (directions[particle] > 0) {
                parallel_count++;
            } else {
                antiparallel_count++;
            }
        }
        
        // Mark the minority group as flipped
        bool mark_parallel_as_flipped = (parallel_count < antiparallel_count);
        for (ParticlePtrType particle : lattice_particles) {
            bool is_parallel = (directions[particle] > 0);
            int particle_index = static_cast<int>(particle - &particles[0]);
            if ((mark_parallel_as_flipped && is_parallel) || (!mark_parallel_as_flipped && !is_parallel)) {
                flipped_results[particle_index] = 1;
            }
        }
    }
}

void get_cleaned_neighbours(float* data, int num_particles, CleanParams* params, int* offsets, int* neighbours_out) {
    g_processor.get_cleaned_neighbours(data, num_particles, params, offsets, neighbours_out);
}

// Function to reset the processor state (useful for testing)
void reset_particles() {
    g_processor.reset();
}

// Access processed particles
const std::vector<Particle>& get_processed_particles() {
    return g_processor.get_particles();
}
