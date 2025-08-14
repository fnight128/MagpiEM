#ifndef PROCESSING_HPP
#define PROCESSING_HPP

#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cmath>
#ifdef _WIN32
    #ifdef BUILDING_DLL
        #define EXPORT __declspec(dllexport)
    #else
        #define EXPORT __declspec(dllimport)
    #endif
#else
    #define EXPORT
#endif

// Maximum number of particles supported (can be increased as needed)
#define MAX_PARTICLES 1000000

struct Particle {
    float x, y, z;
    float rx, ry, rz;
    std::vector<Particle*> neighbors;
    
    // Constructor to initialize from raw float data
    Particle(float x, float y, float z, float rx, float ry, float rz) 
        : x(x), y(y), z(z), rx(rx), ry(ry), rz(rz) {
        neighbors.clear();
    }

    // Default constructor
    Particle() : x(0), y(0), z(0), rx(0), ry(0), rz(0) {
        neighbors.clear();
    }
    
    // Static method to construct particles from raw data
    static std::vector<Particle> from_raw_data(float* data, int num_particles) {
        std::vector<Particle> particles;
        particles.reserve(num_particles);
        
        for (int i = 0; i < num_particles; i++) {
            int base = i * 6; // 6 floats per particle: x, y, z, rx, ry, rz
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

    float calculate_distance_squared(const Particle& p) {
        float dx = x - p.x;
        float dy = y - p.y;
        float dz = z - p.z;
        return dot_product(dx, dy, dz, dx, dy, dz);
    }

    static float dot_product(float x1, float y1, float z1, float x2, float y2, float z2) {
        return x1*x2 + y1*y2 + z1*z2;
    }

    static void normalize_vector(float& x, float& y, float& z) {
        float magnitude = std::sqrt(dot_product(x, y, z, x, y, z));
        if (magnitude != 0.0f) {
            x /= magnitude;
            y /= magnitude;
            z /= magnitude;
        }
    }

    int get_neighbor_count() const {
        return static_cast<int>(neighbors.size());
    }
    
    Particle* get_neighbor(int index) const {
        if (index >= 0 && index < static_cast<int>(neighbors.size())) {
            return neighbors[index];
        }
        throw std::out_of_range("Neighbor index " + std::to_string(index) + 
                               " is out of range [0, " + std::to_string(neighbors.size() - 1) + "]");
    }
    
    void remove_neighbor(Particle* neighbor_to_remove, bool remove_partner = true) {
        auto it = std::find(neighbors.begin(), neighbors.end(), neighbor_to_remove);
        if (it != neighbors.end()) {
            neighbors.erase(it);
            if (remove_partner) {
                neighbor_to_remove->remove_neighbor(this, false);
            }
        } else {
            throw std::invalid_argument("Neighbor not found in particle");
        }
    }
};



#ifdef __cplusplus
extern "C" {
#endif
EXPORT void clean_particles(float* data, int num_points, float min_distance, float max_distance, int* results);
#ifdef __cplusplus
}
#endif

#endif 
