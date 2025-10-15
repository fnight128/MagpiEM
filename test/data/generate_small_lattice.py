#!/usr/bin/env python3
"""
Generate small_test_data.mat by filtering particles from test_data.mat
using small_lattice_coordinates.csv

This provides a simpler dataset which allows for more
straightforward and focussed testing for some functions
"""

import sys
import logging
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Union

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from magpiem.io.io_utils import read_emc_mat, write_emc_mat  # noqa: E402

# Import test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_utils import TestConfig, get_test_data_path  # noqa: E402

logger = logging.getLogger(__name__)

FLIP_PERCENTAGE = 0.3
RANDOM_SEED = 123


def load_coordinates_from_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load particle coordinates from CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file with coordinates

    Returns
    -------
    pd.DataFrame
        DataFrame with particle coordinates
    """
    logger.info(f"Loading coordinates from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} particle coordinates")
    return df


def find_matching_particle_indices(
    geometry_data: Dict[str, np.ndarray],
    coordinates_df: pd.DataFrame,
    tolerance: float = 1e-4,
) -> Dict[str, List[int]]:
    """
    Find particle indices that match the coordinates in coordinates_df.

    Parameters
    ----------
    geometry_data : dict
        Geometry data from the main mat file
    coordinates_df : pd.DataFrame
        DataFrame with target coordinates
    tolerance : float
        Tolerance for coordinate matching

    Returns
    -------
    dict
        Dictionary mapping tomogram names to lists of matching particle indices
    """
    matching_particles = {}

    tomo_name = list(geometry_data.keys())[0]
    logger.info(f"Processing tomogram: {tomo_name}")

    tomo_data = geometry_data[tomo_name]
    tomo_coords = coordinates_df

    logger.debug(f"Tomogram data shape: {tomo_data.shape}")
    logger.debug(f"Target particles: {len(tomo_coords)}")

    if len(tomo_data.shape) == 2 and tomo_data.shape[1] >= 13:
        tomo_x = tomo_data[:, 10]
        tomo_y = tomo_data[:, 11]
        tomo_z = tomo_data[:, 12]

        matching_indices = []

        # For each target coordinate, find the closest match
        for _, target in tomo_coords.iterrows():
            target_x, target_y, target_z = target["x"], target["y"], target["z"]

            # Calculate distances to all particles
            # Highly sub-optimal performance, but only needs to be run once
            distances = np.sqrt(
                (tomo_x - target_x) ** 2
                + (tomo_y - target_y) ** 2
                + (tomo_z - target_z) ** 2
            )

            closest_idx = np.argmin(distances)
            min_distance = distances[closest_idx]

            if min_distance <= tolerance:
                matching_indices.append(closest_idx)
                logger.debug(
                    f"Found match: particle {closest_idx} (distance: {min_distance:.6f})"
                )
            else:
                logger.warning(
                    f"No close match found for target ({target_x}, {target_y}, {target_z})"
                )
                logger.warning(f"Closest distance: {min_distance:.6f}")

        if matching_indices:
            matching_particles[tomo_name] = matching_indices
            logger.info(f"Found {len(matching_indices)} matching particles")
        else:
            logger.warning(f"No matching particles found for {tomo_name}")

    return matching_particles


def create_filtered_mat_file(
    input_mat_path: Union[str, Path],
    coordinates_csv_path: Union[str, Path],
    output_mat_path: Union[str, Path],
) -> bool:
    """
    Create a filtered mat file with only the particles specified in the CSV.

    Parameters
    ----------
    input_mat_path : str or Path
        Path to the input mat file
    coordinates_csv_path : str or Path
        Path to the CSV file with coordinates
    output_mat_path : str or Path
        Path to save the filtered mat file
    """
    logger.info(f"Creating filtered mat file: {output_mat_path}")

    coordinates_df = load_coordinates_from_csv(coordinates_csv_path)

    logger.info(f"Loading main data from {input_mat_path}")
    geometry_data = read_emc_mat(str(input_mat_path))

    if geometry_data is None:
        logger.error("Could not load geometry data from mat file")
        return False

    matching_particles = find_matching_particle_indices(geometry_data, coordinates_df)

    if not matching_particles:
        logger.error("No matching particles found")
        return False

    logger.info(f"Saving filtered data to {output_mat_path}")
    write_emc_mat(
        keep_ids=matching_particles,
        out_path=str(output_mat_path),
        inp_path=str(input_mat_path),
        purge_blanks=True,
    )

    total_particles = sum(len(particles) for particles in matching_particles.values())
    logger.info(f"Created filtered mat file with {total_particles} particles")
    logger.info(f"Output file: {output_mat_path}")

    return True


def get_random_flip_indices(
    matching_particles: Dict[str, List[int]], flip_percentage: float = FLIP_PERCENTAGE
) -> Dict[str, List[int]]:
    """
    Get random particle indices to flip from the matching particles.

    Parameters
    ----------
    matching_particles : dict
        Dictionary mapping tomogram names to lists of particle indices
    flip_percentage : float
        Percentage of particles to flip (default: 0.3 for 30%)

    Returns
    -------
    dict
        Dictionary mapping tomogram names to lists of particle indices to flip
    """
    flip_particles = {}

    for tomo_name, particle_indices in matching_particles.items():
        total_particles = len(particle_indices)
        num_to_flip = int(total_particles * flip_percentage)

        flip_indices = random.sample(particle_indices, num_to_flip)
        flip_particles[tomo_name] = flip_indices

        logger.info(
            f"Tomogram {tomo_name}: {num_to_flip}/{total_particles} particles selected for flipping"
        )

    return flip_particles


def create_flipped_mat_file(
    input_mat_path: Union[str, Path], output_mat_path: Union[str, Path]
) -> bool:
    """
    Create a flipped mat file with randomly selected particles flipped.

    Parameters
    ----------
    input_mat_path : str or Path
        Path to the input mat file
    output_mat_path : str or Path
        Path to save the flipped mat file

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger.info(f"Creating flipped mat file: {output_mat_path}")

    logger.info(f"Loading data from {input_mat_path}")
    geometry_data = read_emc_mat(str(input_mat_path))

    if geometry_data is None:
        logger.error("Could not load geometry data from mat file")
        return False

    matching_particles = _get_all_particle_indices(geometry_data)

    flip_particles = get_random_flip_indices(matching_particles)

    logger.info(f"Saving flipped data to {output_mat_path}")
    write_emc_mat(
        keep_ids=matching_particles,
        out_path=str(output_mat_path),
        inp_path=str(input_mat_path),
        flip_particles=flip_particles,
        purge_blanks=True,
    )

    total_flipped = sum(len(particles) for particles in flip_particles.values())
    total_particles = sum(len(particles) for particles in matching_particles.values())
    flip_percentage_actual = (
        (total_flipped / total_particles * 100) if total_particles > 0 else 0
    )

    logger.info(f"Created flipped mat file with {total_particles} particles")
    logger.info(f"Flipped {total_flipped} particles ({flip_percentage_actual:.1f}%)")
    logger.info(f"Output file: {output_mat_path}")

    return True


def _get_all_particle_indices(
    geometry_data: Dict[str, np.ndarray],
) -> Dict[str, List[int]]:
    """
    Get all particle indices from geometry data.

    Parameters
    ----------
    geometry_data : dict
        Geometry data from emClarity .mat file

    Returns
    -------
    dict
        Dictionary mapping tomogram names to lists of all particle indices
    """
    matching_particles = {}
    for tomo_name, tomo_data in geometry_data.items():
        matching_particles[tomo_name] = list(range(len(tomo_data)))
    return matching_particles


def main():
    """Main function to generate .mat files"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get the directory containing this script
    script_dir = Path(__file__).parent

    # Define file paths relative to the script directory
    input_file = script_dir / TestConfig.TEST_DATA_STANDARD
    coordinates_file = script_dir / TestConfig.TEST_DATA_COORDINATES
    output_file = script_dir / TestConfig.TEST_DATA_SMALL
    flipped_output_file = script_dir / TestConfig.TEST_DATA_SMALL_FLIPPED

    logger.info("Generating test datasets...")
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Coordinates file: {coordinates_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Flipped output file: {flipped_output_file}")

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error(f"Files in script directory: {list(script_dir.iterdir())}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not coordinates_file.exists():
        logger.error(f"Coordinates file not found: {coordinates_file}")
        logger.error(f"Files in script directory: {list(script_dir.iterdir())}")
        raise FileNotFoundError(f"Coordinates file not found: {coordinates_file}")

    # Set seed for reproducible testing
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    logger.info("Creating filtered dataset...")
    success = create_filtered_mat_file(input_file, coordinates_file, output_file)

    if not success:
        logger.error("Failed to generate filtered mat file")
        return

    logger.info(f"Generated {output_file}")

    logger.info("Creating flipped dataset...")
    # flip a subset of particles from the generated filtered dataset
    success = create_flipped_mat_file(output_file, flipped_output_file)

    if not success:
        logger.error("Failed to generate flipped mat file")
        return

    logger.info("Successfully generated test datasets")
    logger.info(f"Created {TestConfig.TEST_DATA_SMALL}")
    logger.info(f"Created {TestConfig.TEST_DATA_SMALL_FLIPPED}")


if __name__ == "__main__":
    main()
