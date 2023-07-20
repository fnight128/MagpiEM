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

from .classes import Tomogram, Particle


def em_format(particle) -> list:
    """
    Format particle data into a list
    formatted for .em files

    Parameters
    ----------
    particle : Particle
        Particle to format

    Returns
    -------
    list
        List formatted for .em file

    """
    rx, ry, rz = particle.direction
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


def purge_blank_tomos(mat_dict: dict, blank_tomos: set) -> dict:
    """
    Recursively purge all mention of blank tomograms from dict.
    mat_dict is modified in place

    Parameters
    ----------
    mat_dict : dict
        subTomoMeta dict from emClarity .mat file
    blank_tomos : set
        Keys of tomograms to remove

    Returns
    -------
    None.

    """
    for blank_tomo in blank_tomos:
        mat_dict.pop(blank_tomo, None)
    for value in mat_dict.values():
        if isinstance(value, dict):
            purge_blank_tomos(value, blank_tomos)


def write_emc_mat(
    keep_ids: dict, out_path, inp_path, purge_blanks: bool = True
) -> None:
    """
    Write a new emClarity .mat database with only the particles defined
    by keep_ids

    Parameters
    ----------
    keep_ids : dict
        IDs of particles to keep
    out_path :
        Path to save new file to
    inp_path :
        Original .mat file to read data from
    purge_blanks : bool, optional
        Whether tomograms with no particles should be removed.
        Defaults to True

    """
    try:
        mat_full = scipy.io.loadmat(inp_path, simplify_cells=True, mat_dtype=True)
        mat_geom = mat_full["subTomoMeta"]["cycle000"]["geometry"]
    except Exception as e:
        print("Unable to open original matlab file, was it moved?")
        raise e
    blank_tomos = set()
    for tomo_id, particles in keep_ids.items():
        if len(particles) == 0:
            blank_tomos.add(tomo_id)
            continue
        table_rows = list()
        mat_table = mat_geom[tomo_id]
        for particle_id in particles:
            table_rows.append(mat_table[particle_id])
        mat_geom[tomo_id] = table_rows
    if purge_blanks and len(blank_tomos) > 0:
        purge_blank_tomos(mat_full, blank_tomos)
    mat_full["subTomoMeta"]["cycle000"]["geometry"] = mat_geom
    scipy.io.savemat(out_path, mdict=mat_full)


def write_emfile(tomo_dict: dict, out_path, out_suffix: str) -> None:
    """
    Write a .em file for each tomo in tomo_dict

    Parameters
    ----------
    tomo_dict : dict
        Tomos to write .em file for
    out_path :
        Directory to write em files
    out_suffix : str
        Suffix to add to each tomo's name to generate filename
        (extension not necessary)

    Returns
    -------
    None.

    """
    for skey, tomo in tomo_dict.items():
        filename = "{0}_{1}{2}".format(skey, out_suffix, ".em")
        good_particles = tomo.get_autocleaned_particles()
        em_list = np.array([[em_format(particle) for particle in good_particles]])
        emfile.write(out_path + filename, em_list, overwrite=True)


def zip_files_with_extension(final_filename: str, extension_to_zip: str):
    """
    Zip all files with a certain extension

    Parameters
    ----------
    final_filename : str
        Desired filename for .zip file
    extension_to_zip : str
        Which extension to zip

    Returns
    -------
    archive_name :
        Zipped file

    """
    archive_name = "{}.zip".format(final_filename)
    filenames = glob("*.{}".format(extension_to_zip))

    with ZipFile(archive_name, mode="w") as zp:
        for filename in filenames:
            zp.write(filename)

    return archive_name


def append_filename(filename: str, suffix: str = "out") -> str:
    """
    Insert a suffix between the end of a filename and its extension

    Parameters
    ----------
    filename : str
        File name
    suffix : str
        Suffix to append. Defaults to "out"

    Returns
    -------
    str
        Modified string

    """
    p = Path(filename)
    return "{0}_{1}{2}".format(p.stem, suffix, p.suffix)


def read_emc_mat(
    filename: str, cycle="cycle000", geom_key="geometry", num_images=-1
) -> dict[str, Tomogram]:
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
    tomograms : dict {string : Tomogram}
        Dictionary of tomograms

    """
    # "geometry" is usually correct for cycle000 but not others
    if cycle != "cycle000" and geom_key == "geometry":
        geom_key = "Avg_geometry"

    try:
        full_geom = scipy.io.loadmat(filename, simplify_cells=True)["subTomoMeta"][
            cycle
        ][geom_key]
    except NotImplementedError:
        # Occurs if file is formatted for older versions of matlab
        print(".mat file is from an old version of matlab. Please update the file.")
        return None

    tomograms = dict()
    total_tomos = len(list(full_geom.keys()))
    for idx, (gkey, geom) in enumerate(full_geom.items()):
        if idx == num_images:
            break
        tomo = Tomogram(gkey)
        pdata = [[row[0], row[10:13], row[22:25]] for row in geom]
        particles = Particle.from_array(pdata, tomo)
        tomo.assign_particles(particles)
        tomograms[gkey] = tomo
        print("Tomo {}/{}: {}".format(idx + 1, total_tomos, gkey))
    return tomograms


def read_relion_star(filename, num_images=-1) -> dict[str, Tomogram]:
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
    ----------
    tomograms : dict {string : Tomogram}
        Dictionary of read tomograms
        If file is unreadable, returns nothing

    """
    full_df = starfile.read(filename, always_dict=True)["particles"]

    tomograms = dict()

    tomo_dfs = full_df.groupby("rlnTomoName")

    for idx, tomo_data in enumerate(tomo_dfs):
        if idx == num_images:
            break
        tomo_name = tomo_data[0]
        tomo_df = tomo_data[1]

        # relion uses preset ids
        ids = list(tomo_df["rlnTomoParticleId"])

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
        tomo = Tomogram(tomo_name)
        particles = Particle.from_array(pdata, tomo, ids=ids)
        tomo.assign_particles(particles)
        tomograms[tomo_name] = tomo

    return tomograms


def write_relion_star(keep_ids: dict, out_path: str, inp_path: str):
    """
    Write a new relion .star file with only the particles defined
    by keep_ids

    Parameters
    ----------
    keep_ids : dict
        IDs of particles to keep
    out_path : str
        Path to save new file to.
    inp_path : str
        Original .star file to read from

    """
    rln_dict = starfile.read(inp_path, always_dict=True)
    full_df = rln_dict["particles"]

    all_wanted_ids = set()

    for ids in keep_ids.values():
        all_wanted_ids.update(ids)

    cleaned_df = full_df[full_df["rlnTomoParticleId"].isin(all_wanted_ids)]

    rln_dict["particles"] = cleaned_df

    starfile.write(rln_dict, out_path)
