# -*- coding: utf-8 -*-
"""
@author: Frank
"""

import MagpiEM
import os

file_path = os.path.realpath(__file__)
print(file_path)

correct_clean_counts = {
    "wt2nd_4004_2": 642,
    "wt2nd_4004_6": 782,
    "wt2nd_4004_7": 869,
}

tomo_dict = MagpiEM.read_emc_mat(
    file_path + "/../test_data/small_gag_set.mat", num_images=3
)
cleaner = MagpiEM.Cleaner(1, 3, 10, 55, 10, 9, 10, 90, 10)

for tomo_id, tomo in tomo_dict.items():
    tomo.set_clean_params(cleaner)
    print(tomo_id)
    tomo.autoclean(vectorised=False)
    print(
        "Non-vectorised",
        [particle.particle_id for particle in tomo.get_auto_cleaned_particles()],
    )
    tomo.reset_cleaning()
    tomo.autoclean(vectorised=True)
    print(
        "Vectorised",
        [particle.particle_id for particle in tomo.get_auto_cleaned_particles()],
    )
    tomo.reset_cleaning()
    tomo.autoclean(vectorised=False)
    print(
        "Non-vectorised again",
        [particle.particle_id for particle in tomo.get_auto_cleaned_particles()],
    )
    assert (
        len(tomo.get_auto_cleaned_particles()) == correct_clean_counts[tomo_id]
    ), "Tomo {} did not return the usual amount of clean particles".format(tomo_id)
    print()
