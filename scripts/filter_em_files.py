import magpiem
import numpy as np
import emfile
from scipy.spatial.transform import Rotation as R


def em_format(particle) -> list:
    rx, ry, rz = particle.orientation
    # PlaceObject uses "zxz" euler angles, but saved in the order "zzx"
    rotation_matrix = np.array([[0, 0, rx], [0, 0, ry], [0, 0, rz]])
    euler = R.from_matrix(rotation_matrix).as_euler("xzx", degrees=True)
    euler_formatted = [euler[0], euler[2], euler[1]]
    return [
        particle.cc_score,
        0.0,
        0,
        0,
        0,
        0,
        0,
        *particle.position,
        0,
        0,
        0,
        0,
        0,
        0,
        *euler_formatted,
        1,
    ]


angs = [0, 20, 40]
ang_inc = 20

dist_range_sq = [100, 10000]

total_particle_count = 0

for ang in angs:
    ang_range = [ang, ang + ang_inc]
    ang_range_dot = [np.cos(np.radians(angle)) for angle in ang_range]
    if ang_range_dot[0] > ang_range_dot[1]:
        ang_range_dot[0], ang_range_dot[1] = ang_range_dot[1], ang_range_dot[0]

    tomo_dict = magpiem.read_emc_mat("nuc_tomo.mat")
    tomo: magpiem.Tomogram = next(iter(tomo_dict.values()))

    tomo.find_particle_neighbours(dist_range_sq)

    filtered_particles = set()

    for particle in tomo.all_particles:
        nn_dist_sq = 99999999
        nn = None
        for neighbour in particle.neighbours:
            neighbour_dist_sq = particle.distance_sq(neighbour)
            if neighbour == particle:
                continue
            elif neighbour_dist_sq > nn_dist_sq:
                continue
            else:
                nn = neighbour
                nn_dist_sq = neighbour_dist_sq
        # check angles
        if not nn:
            continue
        elif ang_range_dot[0] <= particle.dot_orientation(nn) < ang_range_dot[1]:
            filtered_particles.add(nn)
            filtered_particles.add(particle)

    total_particle_count += len(filtered_particles)
    assert len(filtered_particles) > 0, "No particles found in range"

    filename = "nns_{}A.em".format(ang_range)
    em_list = np.array([[em_format(particle) for particle in filtered_particles]])
    emfile.write(filename, em_list, overwrite=True)

    tomo.all_particles = filtered_particles
    magpiem.write_emc_mat({"lamellae_014_1": set(range(len(filtered_particles)))}, "nuc_angles_{}.mat".format(ang_range), "nuc_tomo.mat")

print("final total:", total_particle_count)
