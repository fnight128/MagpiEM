# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:55:11 2022

@author: Frank
"""

import emfile
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.io
from pathlib import Path
from glob import glob
from zipfile import ZipFile
import starfile
import eulerangles

from classes import tomogram, Particle

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
    keep_ids: dict,
    out_path: str,
    inp_path: str,
):
    output_data = dict()
    for tomo_id in keep_ids.keys():
        print(tomo_id)
        table_rows = list()
        try:
            mat_out = scipy.io.loadmat(inp_path, simplify_cells=True, mat_dtype=True)
            mat_inp = mat_out["subTomoMeta"]["cycle000"]["geometry"]
        except:
            print("Unable to open original matlab file, was it moved?")
            return ""
        mat_table = mat_inp[tomo_id]
        for particle_id in keep_ids[tomo_id]:
            table_rows.append(mat_table[particle_id])
        output_data[tomo_id] = table_rows
    try:
        mat_out["subTomoMeta"]["cycle000"]["geometry"] = output_data
        scipy.io.savemat(out_path, mdict=mat_out)
    except:
        print("Unable to save file")


def write_emfile(tomo_dict: dict, out_suffix: str, keep_selected: bool):
    for skey, tomo in tomo_dict.items():
        filename = "{0}_{1}{2}".format(skey, out_suffix, ".em")
        good_particles = tomo.auto_cleaned_particles
        em_list = np.array([[em_format(particle) for particle in good_particles]])
        emfile.write(TEMP_FILE_DIR + filename, em_list, overwrite=True)


def zip_files(final_filename: str, extension_to_zip: str):
    archive_name = "{}.zip".format(final_filename)
    filenames = glob("*.{}".format(extension_to_zip))

    with ZipFile(archive_name, mode="w") as zp:
        for filename in filenames:
            zp.write(filename)

    return archive_name


# def read_imod(filename: str):
#     """


#     Parameters
#     ----------
#     filename : str
#         imod filename

#     Returns
#     -------
#     3xN np array (x, y, z)
#         e.g.
#         [[27.4   99.4   3.6]
#          [301.2  38.2   8.1]
#          ...
#          [43.1   99.0  12.3]
#          [88.2   21.1  33.8]]

#     """
#     imod_inp = imodmodel.read(filename)
#     return np.transpose(np.array([imod_inp[q] for q in ["x", "y", "z"]]))


def append_filename(filename: str, suffix="out"):
    p = Path(filename)
    return "{0}_{1}{2}".format(p.stem, suffix, p.suffix)


def read_emC(filename: str, cycle="cycle000", num_images=-1):
    """
    Read tomograms from a .mat file generated by emClarity

    Parameters
    ----------
    filename : String
        Path to .mat file
    cycle : string, optional
        Key of cycle to read from database
        Defaults to cycle000.
    num_images : int
        Number of tomograms to read
        Defaults to all (-1).

    Returns
    -------
    tomograms : dict {string : tomogram}
        Dictionary of read tomgrams
        If file is unreadable, returns nothing

    """
    geom_key = "geometry" if cycle == "cycle000" else "Avg_geometry"

    try:
        full_geom = scipy.io.loadmat(filename, simplify_cells=True)["subTomoMeta"][
            cycle
        ][geom_key]
    except:
        print("Mat file {} unreadable".format(filename))
        return

    tomograms = dict()
    for idx, (gkey, geom) in enumerate(full_geom.items()):
        if idx == num_images:
            break
        print(gkey)
        tomo = tomogram(gkey)
        pdata = [[row[0], row[10:13], row[22:25]] for row in geom]
        particles = Particle.from_array(pdata, tomo)
        tomo.set_particles(particles)
        # print(tomo.name, len(tomo.all_particles))
        tomograms[gkey] = tomo
    return tomograms


# data=0


def read_relion(filename, num_images=-1):
    """
    Read tomograms from a .star file generated by relion

    Parameters
    ----------
    filename : String
        Path to .mat file
        Defaults to cycle000.
    num_images : int, optional
        Number of tomograms to read
        Defaults to all (-1).

    Returns
    tomograms : dict {string : Tomogram}
        Dictionary of read tomgrams
        If file is unreadable, returns nothing

    """
    try:
        full_df = starfile.read(filename, always_dict=True)["particles"]
    except:
        print("Star file {} unreadable".format(filename))
        return

    tomograms = dict()

    tomo_dfs = full_df.groupby("rlnTomoName")

    for idx, tomo_data in enumerate(tomo_dfs):
        if idx == num_images:
            break
        tomo_name = tomo_data[0]
        tomo_df = tomo_data[1]

        # extract ids
        ids = tomo_df["rlnTomoParticleId"].to_numpy()

        # extract euler angles and convert to z-vectors
        angles = tomo_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]]
        ang_mats = eulerangles.euler2matrix(
            angles, "zyz", intrinsic=True, right_handed_rotation=True
        )
        z_rotation = [ang_mat[2, :] for ang_mat in ang_mats]

        # extract position vectors
        pos = tomo_df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy()

        # extract CC scores (TODO)
        ccs = [10] * len(z_rotation)
        ccs = ccs

        # assemble into standard format
        pdata = [[cc, ps, z_r] for cc, ps, z_r in zip(ccs, pos, z_rotation)]

        # form tomograms
        tomo = tomogram(tomo_name)
        particles = Particle.from_array(pdata, tomo, ids=ids)
        tomo.set_particles(particles)
        tomograms[tomo_name] = tomo

    return tomograms


def modify_relion_star(keep_ids: dict, out_path: str, inp_path: str):
    try:
        rln_dict = starfile.read(inp_path, always_dict=True)
        full_df = rln_dict["particles"]
    except:
        print("Star file {} unreadable".format(inp_path))
        return

    all_wanted_ids = set()

    for ids in keep_ids.values():
        all_wanted_ids.update(ids)

    cleaned_df = full_df[full_df["rlnTomoParticleId"].isin(all_wanted_ids)]

    rln_dict["particles"] = cleaned_df

    starfile.write(rln_dict, out_path)
