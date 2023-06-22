# -*- coding: utf-8 -*-
"""
@author: Frank
"""


import MagpiEM

tomo_dict = MagpiEM.read_emc("./test_data/small_gag_set.mat")
cleaner = MagpiEM.Cleaner(1, 3, 10, 55, 10, 9, 10, 90, 10)


for tomo_id, tomo in tomo_dict.items():
    tomo.set_clean_params(cleaner)
    print(tomo_id)
    tomo.autoclean()
    print(len(tomo.get_auto_cleaned_particles()))


print(len(tomo_dict))

