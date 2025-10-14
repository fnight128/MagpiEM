#!/usr/bin/env python3
"""
Temporary script to create a filtered dataset for testing

Creates a new mat file with the filtered data.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from magpiem.io.io_utils import read_emc_tomogram_raw_data  # noqa: E402
import scipy.io  # noqa: E402

# Configuration
DATA_FILE = project_root / "test" / "data" / "test_data.mat"
OUTPUT_FILE = project_root / "test" / "data" / "test_data_filtered.mat"
TEST_TOMO_NAME = "wt2nd_4004_2"
FILTER_centre = np.array([828, 545, 732])
FILTER_RADIUS = 300.0


def filter_particles_by_position(tomogram_raw_data, centre, radius):
    """
    Filter particles to only include those within radius of centre position.

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]
    centre : np.array
        centre position [x, y, z]
    radius : float
        Maximum distance from centre

    Returns
    -------
    tuple
        (filtered_data, filtered_indices) where filtered_indices are the original indices
    """
    filtered_data = []
    filtered_indices = []

    for i, particle in enumerate(tomogram_raw_data):
        position = np.array(particle[0])  # [x, y, z]
        distance = np.linalg.norm(position - centre)

        if distance <= radius:
            filtered_data.append(particle)
            filtered_indices.append(i)

    return filtered_data, filtered_indices


def main():
    """Main function to create filtered dataset."""
    print(f"Loading data from {DATA_FILE}")

    mat_full = scipy.io.loadmat(str(DATA_FILE), simplify_cells=True, mat_dtype=True)
    mat_geom = mat_full["subTomoMeta"]["cycle000"]["geometry"]

    print(f"Loaded geometry data with {len(mat_geom)} tomograms")

    if TEST_TOMO_NAME not in mat_geom:
        print(f"Error: Tomogram {TEST_TOMO_NAME} not found in the data")
        return

    tomo_raw_data = read_emc_tomogram_raw_data(mat_geom[TEST_TOMO_NAME], TEST_TOMO_NAME)
    if tomo_raw_data is None:
        print(f"Error: Failed to extract data for tomogram: {TEST_TOMO_NAME}")
        return

    print(f"Original dataset has {len(tomo_raw_data)} particles")

    print(f"Filtering particles within {FILTER_RADIUS} units of {FILTER_centre}")
    filtered_data, filtered_indices = filter_particles_by_position(
        tomo_raw_data, FILTER_centre, FILTER_RADIUS
    )

    print(f"Filtered dataset has {len(filtered_data)} particles")
    print(
        f"Filtered indices: {filtered_indices[:10]}{'...' if len(filtered_indices) > 10 else ''}"
    )

    if len(filtered_data) == 0:
        print("Error: No particles found within the specified radius")
        return

    original_tomo_data = mat_geom[TEST_TOMO_NAME]
    filtered_tomo_data = []

    for idx in filtered_indices:
        filtered_tomo_data.append(original_tomo_data[idx])

    new_mat_full = mat_full.copy()
    new_mat_full["subTomoMeta"]["cycle000"]["geometry"] = {
        TEST_TOMO_NAME: filtered_tomo_data
    }

    print(f"Saving filtered dataset to {OUTPUT_FILE}")
    scipy.io.savemat(str(OUTPUT_FILE), mdict=new_mat_full)

    print("Filtered dataset saved successfully!")
    print(f"  - Original particles: {len(tomo_raw_data)}")
    print(f"  - Filtered particles: {len(filtered_data)}")
    print(f"  - Reduction: {(1 - len(filtered_data) / len(tomo_raw_data)) * 100:.1f}%")

    positions = np.array([particle[0] for particle in filtered_data])
    print("  - Position range:")
    print(f"    X: {positions[:, 0].min():.1f} to {positions[:, 0].max():.1f}")
    print(f"    Y: {positions[:, 1].min():.1f} to {positions[:, 1].max():.1f}")
    print(f"    Z: {positions[:, 2].min():.1f} to {positions[:, 2].max():.1f}")


if __name__ == "__main__":
    main()
