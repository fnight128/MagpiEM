# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:55:11 2022

@author: Frank
"""

import emfile
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.io
from pathlib import Path
from glob import glob
from zipfile import ZipFile
import starfile

import eulerangles

from .tomogram import Tomogram
from .particle import Particle

logger = logging.getLogger(__name__)


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
        logger.error("Unable to open original matlab file, was it moved?")
        raise e

    logger.info(f"Processing {len(keep_ids)} tomograms for saving")

    tomograms_with_particles = set(keep_ids.keys())

    new_geom = {}

    for tomo_id, particles in keep_ids.items():
        if len(particles) == 0:
            logger.warning(f"Skipping {tomo_id} (no particles)")
            continue

        logger.info(f"Processing {tomo_id} with {len(particles)} particles")
        table_rows = list()
        mat_table = mat_geom[tomo_id]
        for particle_id in particles:
            table_rows.append(mat_table[particle_id])
        new_geom[tomo_id] = table_rows

    # Replace geometry and all other fields with only tomograms that have particles
    mat_full["subTomoMeta"]["cycle000"]["geometry"] = new_geom
    sub_tomo = mat_full["subTomoMeta"]

    # Function to filter metadata sections
    def filter_metadata_section(section_data):
        """Filter a metadata section to only include tomograms with particles."""
        if isinstance(section_data, dict):
            return {
                tomo_id: data
                for tomo_id, data in section_data.items()
                if tomo_id in tomograms_with_particles
            }
        else:
            # For list/array types, filter based on tomogram names
            return [item for item in section_data if item in tomograms_with_particles]

    # Apply function to metadata
    for section_name in ["ctfGroupSize", "reconGeometry", "tiltGeometry"]:
        if section_name in sub_tomo:
            sub_tomo[section_name] = filter_metadata_section(sub_tomo[section_name])

    # mapBackGeometry may not exist
    if "mapBackGeometry" in sub_tomo:
        map_back = sub_tomo["mapBackGeometry"]
        if "tomoName" in map_back:
            map_back["tomoName"] = filter_metadata_section(map_back["tomoName"])

    logger.info(f"Final geometry contains {len(new_geom)} tomograms")
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


def read_multiple_tomograms(
    filename: str, num_images=-1, **kwargs
) -> dict[str, Tomogram] | None:
    """
    Read tomograms from a .mat file generated by emClarity

    Parameters
    ----------
    filename : String
        Path to .mat file
    num_images : int, optional
        Number of tomograms to read
        Defaults to all (-1).
    geom_key: str, optional
        Name of geometry within .mat file
    cycle : string, optional
        Key of cycle to read from database
        Defaults to cycle000.

    Returns
    -------
    tomograms : dict {string : Tomogram}
        Dictionary of tomograms

    """
    full_geom = read_emc_mat(filename, **kwargs)
    if full_geom is None:
        return None

    tomograms = dict()
    total_tomos = len(list(full_geom.keys()))
    for idx, (gkey, geom) in enumerate(full_geom.items()):
        if idx == num_images:
            break
        tomograms[gkey] = read_emc_tomogram(geom, gkey)
        logger.debug("Tomo {}/{}: {}".format(idx + 1, total_tomos, gkey))
    return tomograms


def read_single_tomogram(filename: str, tomogram_name: str, **kwargs) -> Tomogram:
    """Read single tomogram"""
    tomo_geom = read_emc_mat(filename, **kwargs)[tomogram_name]
    return read_emc_tomogram(tomo_geom, tomogram_name)


def read_tomo_names(filename: str) -> list:
    return list(read_emc_mat(filename).keys())


def read_emc_mat(filename: str, cycle="cycle000", geom_key="geometry") -> dict | None:
    """
    Read emClarity .mat file into dictionary of tomograms
    Parameters
    ----------
    filename
        .mat file to read
    geom_key: str, optional
        Name of geometry within .mat file
    cycle : string, optional
        Key of cycle to read from database
        Defaults to cycle000.

    Returns
    -------
    geometry : dict {string : Tomogram}
        Dictionary of tomograms in emC format

    """
    # "geometry" is usually correct for cycle000 but not others
    if cycle != "cycle000" and geom_key == "geometry":
        geom_key = "Avg_geometry"

    try:
        full_geom = scipy.io.loadmat(filename, simplify_cells=True)["subTomoMeta"][
            cycle
        ][geom_key]
    except NotImplementedError as e:
        # Occurs if file is formatted for older versions of matlab
        logger.error(
            ".mat file is from an old version of matlab. Please update the file."
        )
        raise e

    return full_geom


def read_emc_tomogram(tomogram_geometry: dict, tomo_name: str) -> Tomogram:
    """
    Extract particle data from emc-formatted tomogram
    Parameters
    ----------
    tomogram_geometry
    tomo_name

    Returns
    -------

    """
    tomo = Tomogram(tomo_name)
    pdata = [[row[0], row[10:13], row[22:25]] for row in tomogram_geometry]
    particles = Particle.from_array(pdata, tomo)
    tomo.assign_particles(particles)
    return tomo


def read_emc_tomogram_raw_data(tomogram_geometry: dict, tomo_name: str) -> list:
    """
    Extract raw particle data from emc-formatted tomogram
    Parameters
    ----------
    tomogram_geometry : dict
        Geometry data from emClarity .mat file
    tomo_name : str
        Name of the tomogram

    Returns
    -------
    list
        2D list of raw data for particles in tomo.
        Each row is a particle, in the form [[x,y,z], [rx,ry,rz]]
        Orientations are normalised to unit vectors.
    """
    import numpy as np

    raw_data = [[row[10:13], row[22:25]] for row in tomogram_geometry]

    # Normalise orientations to unit vectors
    for particle in raw_data:
        position, orientation = particle
        # Convert to numpy array for vector operations
        orient_array = np.array(orientation)
        # Normalise to unit vector
        norm = np.linalg.norm(orient_array)
        if norm > 0:  # Avoid division by zero
            particle[1] = (orient_array / norm).tolist()

    return raw_data


def load_single_tomogram_raw_data(
    filename: str, tomogram_name: str, **kwargs
) -> list | None:
    """
    Load raw data for a single tomogram from a .mat file.

    Parameters
    ----------
    filename : str
        Path to .mat file
    tomogram_name : str
        Name of the specific tomogram to load
    geom_key: str, optional
        Name of geometry within .mat file
    cycle : string, optional
        Key of cycle to read from database
        Defaults to cycle000.

    Returns
    -------
    raw_data : list
        Raw tomogram data for the specified tomogram
    """
    full_geom = read_emc_mat(filename, **kwargs)
    if full_geom is None or tomogram_name not in full_geom:
        return None

    return read_emc_tomogram_raw_data(full_geom[tomogram_name], tomogram_name)


def get_tomogram_names(filename: str, num_images=-1, **kwargs) -> list[str] | None:
    """
    Get tomogram names from a .mat file without returning all the data.

    Parameters
    ----------
    filename : str
        Path to .mat file
    num_images : int, optional
        Number of tomograms to get names for
        Defaults to all (-1).
    geom_key: str, optional
        Name of geometry within .mat file
    cycle : string, optional
        Key of cycle to read from database
        Defaults to cycle000.

    Returns
    -------
    tomogram_names : list[str]
        List of tomogram names
    """
    full_geom = read_emc_mat(filename, **kwargs)
    if full_geom is None:
        return None

    tomogram_names = list(full_geom.keys())
    if num_images > 0:
        tomogram_names = tomogram_names[:num_images]

    return tomogram_names


def read_multiple_tomograms_raw_data(
    filename: str, num_images=-1, **kwargs
) -> dict[str, list] | None:
    """
    Read raw tomogram data from a .mat file generated by emClarity

    Parameters
    ----------
    filename : str
        Path to .mat file
    num_images : int, optional
        Number of tomograms to read
        Defaults to all (-1).
    geom_key: str, optional
        Name of geometry within .mat file
    cycle : string, optional
        Key of cycle to read from database
        Defaults to cycle000.

    Returns
    -------
    tomogram_data : dict {string : list}
        Dictionary of raw tomogram data

    """
    full_geom = read_emc_mat(filename, **kwargs)
    if full_geom is None:
        return None

    tomogram_data = dict()
    total_tomos = len(list(full_geom.keys()))
    for idx, (gkey, geom) in enumerate(full_geom.items()):
        if idx == num_images:
            break
        tomogram_data[gkey] = read_emc_tomogram_raw_data(geom, gkey)
        logger.debug("Tomo {}/{}: {}".format(idx + 1, total_tomos, gkey))
    return tomogram_data


def read_relion_star(filename, num_images=-1) -> dict[str, Tomogram]:
    """
    Read tomograms from a .star file generated by relion

    Parameters
    ----------
    filename : String
        Path to .mat file
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
