#ifndef PROCESSING_HPP
#define PROCESSING_HPP

#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cmath>
#include <array>
#ifdef _WIN32
    #ifdef BUILDING_DLL
        #define EXPORT __declspec(dllexport)
    #else
        #define EXPORT __declspec(dllimport)
    #endif
#else
    #define EXPORT
#endif

#define MAX_PARTICLES 1000000

struct CleanParams {
    float min_distance;
    float max_distance;
    float min_orientation;
    float max_orientation;
    float min_curvature;
    float max_curvature;
    int min_lattice_size;
    int min_neighbours;
    
    CleanParams(float min_dist, float max_dist,
                float min_ori, float max_ori,
                float min_curv, float max_curv,
                int min_lattice_size = 10, int min_neigh = 3)
        : min_distance(min_dist), max_distance(max_dist),
          min_orientation(min_ori), max_orientation(max_ori),
          min_curvature(min_curv), max_curvature(max_curv),
          min_lattice_size(min_lattice_size), min_neighbours(min_neigh) {}
};

struct Particle {
    float position[3];    // x, y, z
    float orientation[3]; // rx, ry, rz
    int lattice;
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

    float calculate_distance_squared(const Particle& p) const {
        float dx = position[0] - p.position[0];
        float dy = position[1] - p.position[1];
        float dz = position[2] - p.position[2];
        return dx*dx + dy*dy + dz*dz;
    }

    static float dot_product(const float* vec1, const float* vec2) {
        return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    }

    static void normalize_vec3(float* vec) {
        float magnitude = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
        if (magnitude > 0.0f) {
            float inv_magnitude = 1.0f / magnitude;
            vec[0] *= inv_magnitude;
            vec[1] *= inv_magnitude;
            vec[2] *= inv_magnitude;
        }
        else {
            
        }
    }

    float curvature(const Particle& p) const {
        std::array<float, 3> displacement = displacement_vector(p);
        normalize_vec3(displacement.data());
        return dot_product(displacement.data(), p.orientation);
    }

    int get_neighbour_count() const {
        return static_cast<int>(neighbours.size());
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
EXPORT void clean_particles(float* data, int num_points, CleanParams* params, int* results);
EXPORT void find_neighbours(float* data, int num_points, float min_distance, float max_distance, int* results);
EXPORT void filter_by_orientation(float* data, int num_points, float min_orientation, float max_orientation, int* results);
EXPORT void filter_by_curvature(float* data, int num_points, float min_curvature, float max_curvature, int* results);
#ifdef __cplusplus
}
#endif

#endif 
