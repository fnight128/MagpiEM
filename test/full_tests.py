# -*- coding: utf-8 -*-
"""
@author: Frank
"""

import MagpiEM

correct_clean_counts = {
    "wt2nd_4004_2": 642,
    "wt2nd_4004_6": 782,
    "wt2nd_4004_7": 869,
}

tomo_dict = MagpiEM.read_emc_mat("./test_data/small_gag_set.mat", num_images=3)
cleaner = MagpiEM.Cleaner(1, 3, 10, 55, 10, 9, 10, 90, 10)


for tomo_id, tomo in tomo_dict.items():
    tomo.set_clean_params(cleaner)
    print(tomo_id)
    tomo.autoclean()
    print(len(tomo.get_auto_cleaned_particles()))
    assert (
        len(tomo.get_auto_cleaned_particles()) == correct_clean_counts[tomo_id]
    ), "Tomo {} did not return the usual amount of clean particles".format(tomo_id)
    print()
