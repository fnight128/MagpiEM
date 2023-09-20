import magpiem
import numpy as np
import pandas as pd

tomo_dict = magpiem.read_emc_mat("nuc_tomo.mat")

tomo: magpiem.Tomogram = next(iter(tomo_dict.values()))

tomo.find_particle_neighbours((90, 13000))

nn_params = []

for particle in tomo.all_particles:
    if len(particle.neighbours) == 0: continue
    nn_dist_sq = 99999999
    nn_ang = 0
    for neighbour in particle.neighbours:
        dist = particle.distance_sq(neighbour)
        if particle.distance_sq(neighbour) < nn_dist_sq:
            nn_dist_sq = dist
            nn_ang = particle.dot_orientation(neighbour)
    nn_params.append((nn_dist_sq ** 0.5, np.degrees(np.arccos(nn_ang))))

df = pd.DataFrame(nn_params, columns=["Distance", "Orientation"])
df.to_csv("neighbour_params.csv")
