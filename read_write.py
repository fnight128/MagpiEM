# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:55:11 2022

@author: Frank
"""

import imodmodel
import emfile
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.io
from pathlib import Path

MAT_KEY = "cycle000"

if MAT_KEY == "cycle000":
    MAT_KEY2 = "geometry"
else:
    MAT_KEY2 = "Avg_geometry"

# readin
TEMP_FILE_DIR = "static/"


def em_format(particle):
    rx, ry, rz = particle.direction
    # convert orientation vector into euler angles
    # PlaceObject uses "zxz" euler angles, but saved in the order "zzx"
    rotation_matrix = np.array([[0, 0, rx], [0, 0, ry], [0, 0, rz]])
    euler = R.from_matrix(rotation_matrix).as_euler("xzx", degrees=True)
    euler_formatted = [euler[0], euler[2], euler[1]]
    # print("")
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


def modify_emc_mat(
    subtomo_dict: dict, out_path: str, inp_path: str, keep_selected: bool
):
    output_data = dict()
    for tomo_id in subtomo_dict.keys():
        print(tomo_id)
        table_rows = list()
        try:
            mat_out = scipy.io.loadmat(inp_path, simplify_cells=True)
            mat_inp = mat_out["subTomoMeta"][MAT_KEY][MAT_KEY2]
        except:
            print("Unable to open original matlab file, was it moved?")
            return ""
        mat_table = mat_inp[tomo_id]
        if keep_selected:
            for particle_id in subtomo_dict[tomo_id].selected_particle_ids():
                table_rows.append(mat_table[particle_id])
        else:
            for particle_id in subtomo_dict[tomo_id].unselected_particle_ids():
                table_rows.append(mat_table[particle_id])
        output_data[tomo_id] = table_rows

    try:
        mat_out["subTomoMeta"][MAT_KEY][MAT_KEY2] = output_data
        scipy.io.savemat(out_path, mdict=mat_out)
    except:
        print("Unable to save file")


def write_emfile(subtomo_dict: dict, out_suffix: str, keep_selected: bool):
    for skey, tomo in subtomo_dict.items():
        filename = "{0}_{1}{2}".format(skey, out_suffix, ".em")
        good_particles = tomo.auto_cleaned_particles
        em_list = np.array([[em_format(particle) for particle in good_particles]])
        emfile.write(TEMP_FILE_DIR + filename, em_list, overwrite=True)


def read_imod(filename: str):
    """


    Parameters
    ----------
    filename : str
        imod filename

    Returns
    -------
    3xN np array (x, y, z)
        e.g.
        [[27.4   99.4   3.6]
         [301.2  38.2   8.1]
         ...
         [43.1   99.0  12.3]
         [88.2   21.1  33.8]]

    """
    imod_inp = imodmodel.read(filename)
    return np.transpose(np.array([imod_inp[q] for q in ["x", "y", "z"]]))


def read_emfile(filename: str):
    header, em_inp = emfile.read(filename)
    print(em_inp)


def append_filename(filename, suffix="out"):
    p = Path(filename)
    return "{0}_{1}{2}".format(p.stem, suffix, p.suffix)


# data=0


# print("header", header)
