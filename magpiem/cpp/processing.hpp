#ifndef PROCESSING_HPP
#define PROCESSING_HPP

#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cmath>
#include <array>
#include <immintrin.h> // For AVX/SSE intrinsics

#define MAX_PARTICLES 1000000

struct CleanParams {
    float min_distance;
    float max_distance;
    float min_orientation;
    float max_orientation;
    float min_curvature;
    float max_curvature;
    unsigned int min_lattice_size;
    unsigned int min_neighbours;
    bool allow_flips;
    
    CleanParams(float min_dist, float max_dist,
                float min_ori, float max_ori,
                float min_curv, float max_curv,
                unsigned int min_lattice_size = 10, unsigned int min_neigh = 3,
                bool allow_flips = false)
        : min_distance(min_dist), max_distance(max_dist),
          min_orientation(min_ori), max_orientation(max_ori),
          min_curvature(min_curv), max_curvature(max_curv),
          min_lattice_size(min_lattice_size), min_neighbours(min_neigh),
          allow_flips(allow_flips) {}
};

struct Particle {
    float position[3];    // x, y, z
    float orientation[3]; // rx, ry, rz
    unsigned int lattice;
    std::vector<Particle*> neighbours;
    
    Particle(float x, float y, float z, float rx, float ry, float rz) {
        position[0] = x;
        position[1] = y;
        position[2] = z;
        orientation[0] = rx;
        orientation[1] = ry;
        orientation[2] = rz;
        lattice = 0;
        normalize_vec3(orientation);
        neighbours.clear();
    }

    Particle() {
        position[0] = position[1] = position[2] = 0.0f;
        orientation[0] = orientation[1] = orientation[2] = 0.0f;
        lattice = 0;
        neighbours.clear();
    }
    
    static std::vector<Particle> from_raw_data(float* data, int num_particles) {
        std::vector<Particle> particles;
        particles.reserve(num_particles);
        
        for (int i = 0; i < num_particles; i++) {
            int base = i * 6;
            particles.emplace_back(
                data[base],     // x
                data[base + 1], // y
                data[base + 2], // z
                data[base + 3], // rx
                data[base + 4], // ry
                data[base + 5]  // rz
            );
        }
        return particles;
    }

    std::array<float, 3> displacement_vector(const Particle& p) const {
        std::array<float, 3> displacement;
        displacement[0] = position[0] - p.position[0];
        displacement[1] = position[1] - p.position[1];
        displacement[2] = position[2] - p.position[2];
        return displacement;
    }

    // Helper function to extract x, y, z components from SSE vector
    static void extract_sse_components(const __m128& vec, float& x, float& y, float& z) {
        x = _mm_cvtss_f32(vec);
        y = _mm_cvtss_f32(_mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1)));
        z = _mm_cvtss_f32(_mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2)));
    }

    float calculate_distance_squared(const Particle& p) const {
        // Vectorised distance calculation using SSE
        __m128 pos1 = _mm_loadu_ps(position);
        __m128 pos2 = _mm_loadu_ps(p.position);
        __m128 diff = _mm_sub_ps(pos1, pos2);
        __m128 squared = _mm_mul_ps(diff, diff);
        
        // Extract individual components using helper
        float dx2, dy2, dz2;
        extract_sse_components(squared, dx2, dy2, dz2);
        
        return dx2 + dy2 + dz2;
    }

    static float dot_product(const float* vec1, const float* vec2) {
        // Vectorised dot product using SSE
        __m128 v1 = _mm_loadu_ps(vec1);
        __m128 v2 = _mm_loadu_ps(vec2);
        __m128 product = _mm_mul_ps(v1, v2);
        
        // Extract individual components using helper
        float x, y, z;
        extract_sse_components(product, x, y, z);
        
        return x + y + z;
    }

    static void normalize_vec3(float* vec) {
        // Vectorised normalisation using SSE
        __m128 v = _mm_loadu_ps(vec);
        __m128 squared = _mm_mul_ps(v, v);
        
        // Extract individual components using helper
        float x2, y2, z2;
        extract_sse_components(squared, x2, y2, z2);
        
        float mag = std::sqrt(x2 + y2 + z2);
        
        if (mag > 0.0f) {
            float inv_magnitude = 1.0f / mag;
            __m128 inv_mag_vec = _mm_set1_ps(inv_magnitude);
            __m128 normalized = _mm_mul_ps(v, inv_mag_vec);
            _mm_storeu_ps(vec, normalized);
        }
        else {
            throw std::invalid_argument("Attempted to normalise a zero vector");
        }
    }

    float curvature(const Particle& p) const {
        std::array<float, 3> displacement = displacement_vector(p);
        normalize_vec3(displacement.data());
        return dot_product(displacement.data(), p.orientation);
    }

    unsigned int get_neighbour_count() const {
        return static_cast<unsigned int>(neighbours.size());
    }

    Particle* get_neighbour(int index) const {
        if (index >= 0 && index < static_cast<int>(neighbours.size())) {
            return neighbours[index];
        }
        throw std::out_of_range("Neighbour index " + std::to_string(index) + 
                               " is out of range [0, " + std::to_string(neighbours.size() - 1) + "]");
    }

    void remove_neighbour(Particle* neighbour_to_remove, bool remove_partner = true) {
        auto it = std::find(neighbours.begin(), neighbours.end(), neighbour_to_remove);
        if (it != neighbours.end()) {
            neighbours.erase(it);
            if (remove_partner) {
                neighbour_to_remove->remove_neighbour(this, false);
            }
        } else {
            throw std::invalid_argument("Neighbour not found in particle");
        }
    }
    

};

#ifdef __cplusplus
extern "C" {
#endif
void clean_particles(float* data, int num_points, CleanParams* params, int* results);
void find_neighbours(float* data, int num_points, float min_distance, float max_distance, int* results);
void filter_by_orientation(float* data, int num_points, float min_orientation, float max_orientation, bool allow_flips, int* results);
void filter_by_curvature(float* data, int num_points, float min_curvature, float max_curvature, int* results);
void assign_lattices(float* data, int num_points, unsigned int min_neighbours, unsigned int min_lattice_size, int* results);
// Debug/testing utility: perform distance + orientation + curvature filtering and
// return neighbour lists in CSR form. Offsets has length num_points + 1. If
// neighbours_out is nullptr, only offsets are filled (offsets[num_points] will be total entries).
void get_cleaned_neighbours(float* data, int num_points, CleanParams* params, int* offsets, int* neighbours_out);
// Combined cleaning and flip detection: returns lattice assignments and flipped particle flags
void clean_and_detect_flips(float* data, int num_points, CleanParams* params, int* lattice_results, int* flipped_results);
// Debug function for testing flip detection with manual lattice assignment
void debug_flip_detection(float* data, int num_points, CleanParams* params, int* lattice_results, int* flipped_results) noexcept(false);

// Function to get access to processed particles (for flip detection)
const std::vector<Particle>& get_processed_particles();

void set_log_level(int level);
#ifdef __cplusplus
}
#endif

#endif 
