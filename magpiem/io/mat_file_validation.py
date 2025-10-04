#!/usr/bin/env python3
"""
Test script to compare two .mat files and verify they have the same structure
and values, only differing in the number of particles and/or tomograms.
"""

import scipy.io
import numpy as np
import os
import sys
import logging
from typing import Dict, Any, List

log = logging.getLogger(__name__)

TOMOGRAMS_TO_DISPLAY = 5

# Items where changes are expected
# Header contains information on creation date, so will always be different
# Other categories will have some particles removed, and potentially
# entire tomograms, too
IGNORED_ITEMS = {
    "root": ["__header__"],
    "root.subTomoMeta.cycle000": ["geometry"],
    "root.subTomoMeta": ["ctfGroupSize", "tiltGeometry", "reconGeometry"],
    "root.subTomoMeta.mapBackGeometry": ["tomoName"],
}


def load_mat_file(filepath: str) -> Dict[str, Any]:
    """Load a .mat file and return its contents."""
    try:
        return scipy.io.loadmat(filepath, simplify_cells=True, mat_dtype=True)
    except Exception as e:
        raise ValueError(f"Error loading {filepath}: {e}")


def get_file_info(filepath: str) -> Dict[str, Any]:
    """Get basic file information."""
    return {
        "path": filepath,
        "size_mb": os.path.getsize(filepath) / (1024 * 1024),
        "exists": os.path.exists(filepath),
    }


def extract_geometry_info(mat_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract geometry information from .mat file."""
    if "subTomoMeta" not in mat_data:
        return {"error": "No subTomoMeta found"}

    sub_tomo = mat_data["subTomoMeta"]
    if "cycle000" not in sub_tomo:
        return {"error": "No cycle000 found"}

    cycle = sub_tomo["cycle000"]
    if "geometry" not in cycle:
        return {"error": "No geometry found"}

    geometry = cycle["geometry"]

    # Count tomograms and particles
    tomogram_info = {}
    total_particles = 0

    for tomo_name, tomo_data in geometry.items():
        if isinstance(tomo_data, (list, np.ndarray)):
            particle_count = len(tomo_data)
        else:
            particle_count = 0
        tomogram_info[tomo_name] = particle_count
        total_particles += particle_count

    return {
        "tomogram_count": len(geometry),
        "total_particles": total_particles,
        "tomogram_details": tomogram_info,
        "geometry_keys": list(geometry.keys()),
    }


def should_ignore_item(path: str, key: str) -> bool:
    """Check if an item should be ignored during comparison."""
    return path in IGNORED_ITEMS and key in IGNORED_ITEMS[path]


def compare_arrays(val1: Any, val2: Any, path: str, key: str) -> List[str]:
    """Compare array-like objects and return differences."""
    differences = []

    # Skip content comparison for geometry arrays
    if path == "root.subTomoMeta.cycle000.geometry":
        return differences

    if len(val1) != len(val2):
        differences.append(
            f"{path}.{key}: Different lengths - {len(val1)} vs {len(val2)}"
        )
    elif len(val1) > 0 and len(val2) > 0:
        sample_size = min(3, len(val1), len(val2))
        for i in range(sample_size):
            if isinstance(val1[i], (list, np.ndarray)) and isinstance(
                val2[i], (list, np.ndarray)
            ):
                if len(val1[i]) != len(val2[i]):
                    differences.append(
                        f"{path}.{key}[{i}]: Different sub-array lengths - {len(val1[i])} vs {len(val2[i])}"
                    )
            elif val1[i] != val2[i]:
                differences.append(
                    f"{path}.{key}[{i}]: Different values - {val1[i]} vs {val2[i]}"
                )

    return differences


def compare_mat_structures(
    mat1: Dict[str, Any], mat2: Dict[str, Any], path: str = "root"
) -> List[str]:
    """Recursively compare two .mat file structures and return differences."""
    differences = []

    keys1 = set(mat1.keys())
    keys2 = set(mat2.keys())

    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1

    if missing_in_2:
        differences.append(f"{path}: Keys missing in file 2: {missing_in_2}")
    if missing_in_1:
        differences.append(f"{path}: Keys missing in file 1: {missing_in_1}")

    common_keys = keys1 & keys2
    for key in common_keys:
        if should_ignore_item(path, key):
            continue

        val1 = mat1[key]
        val2 = mat2[key]

        if type(val1) != type(val2):
            differences.append(
                f"{path}.{key}: Different types - {type(val1)} vs {type(val2)}"
            )
            continue

        if isinstance(val1, dict):
            sub_diffs = compare_mat_structures(val1, val2, f"{path}.{key}")
            differences.extend(sub_diffs)
        elif isinstance(val1, (list, np.ndarray)):
            differences.extend(compare_arrays(val1, val2, path, key))
        else:
            if val1 != val2:
                differences.append(f"{path}.{key}: Different values - {val1} vs {val2}")

    return differences


def compare_particle_data_structure(
    mat1: Dict[str, Any], mat2: Dict[str, Any]
) -> List[str]:
    """Compare the structure of particle data between two .mat files."""
    differences = []

    try:
        geom1 = mat1["subTomoMeta"]["cycle000"]["geometry"]
        geom2 = mat2["subTomoMeta"]["cycle000"]["geometry"]
    except KeyError as e:
        differences.append(f"Missing key in structure: {e}")
        return differences

    common_tomos = set(geom1.keys()) & set(geom2.keys())

    if not common_tomos:
        differences.append("No common tomograms found for structure comparison")
        return differences

    # Compare structure of first common tomogram
    sample_tomo = list(common_tomos)[0]
    particles1 = geom1[sample_tomo]
    particles2 = geom2[sample_tomo]

    if len(particles1) == 0 or len(particles2) == 0:
        differences.append(
            f"Tomogram {sample_tomo} has no particles in one or both files"
        )
        return differences

    # Compare particle data structure
    particle1 = particles1[0]
    particle2 = particles2[0]

    # Handle case where particle data might be scalar values
    if hasattr(particle1, "__len__") and hasattr(particle2, "__len__"):
        if len(particle1) != len(particle2):
            differences.append(
                f"Different particle data lengths: {len(particle1)} vs {len(particle2)}"
            )
        else:
            for i, (val1, val2) in enumerate(zip(particle1, particle2)):
                if type(val1) != type(val2):
                    differences.append(
                        f"Particle field {i}: Different types - {type(val1)} vs {type(val2)}"
                    )
    else:
        # Handle scalar particle data
        if type(particle1) != type(particle2):
            differences.append(
                f"Particle data: Different types - {type(particle1)} vs {type(particle2)}"
            )

    return differences


def log_file_info(info: Dict[str, Any], file_num: int):
    """Print file information."""
    log.info(f"File {file_num}: {info['path']}")
    log.info(f"  Size: {info['size_mb']:.2f} MB")
    log.info(f"  Exists: {info['exists']}")
    log.info("")


def log_geometry_info(geom_info: Dict[str, Any], file_num: int):
    """Print geometry information."""
    log.info(f"File {file_num} Geometry:")
    log.info(f"  Tomograms: {geom_info.get('tomogram_count', 'N/A')}")
    log.info(f"  Total particles: {geom_info.get('total_particles', 'N/A')}")
    if "tomogram_details" in geom_info:
        log.info(
            f"  First {TOMOGRAMS_TO_DISPLAY} tomograms: {list(geom_info['tomogram_details'].items())[:TOMOGRAMS_TO_DISPLAY]}"
        )
    log.info("")


def log_comparison_results(structure_diffs: List[str], particle_diffs: List[str]):
    """Print comparison results."""
    # Compare structures
    log.info("Comparing file structures...")
    if structure_diffs:
        log.warning("STRUCTURE DIFFERENCES FOUND:")
        for diff in structure_diffs:
            log.warning(f"  - {diff}")
        log.info("")
    else:
        log.info("✓ File structures are identical (excluding geometry content)")
        log.info("")

    # Compare particle data structure
    log.info("Comparing particle data structure...")
    if particle_diffs:
        log.warning("PARTICLE DATA DIFFERENCES FOUND:")
        for diff in particle_diffs:
            log.warning(f"  - {diff}")
        log.info("")
    else:
        log.info("✓ Particle data structures are identical")
        log.info("")


def log_summary(
    info1: Dict[str, Any],
    info2: Dict[str, Any],
    structure_diffs: List[str],
    particle_diffs: List[str],
):
    """Print summary of comparison."""
    log.info("=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)

    size_diff = abs(info1["size_mb"] - info2["size_mb"])
    size_ratio = max(info1["size_mb"], info2["size_mb"]) / min(
        info1["size_mb"], info2["size_mb"]
    )

    log.info(f"Size difference: {size_diff:.2f} MB (ratio: {size_ratio:.2f}x)")
    log.info(f"Structure differences: {len(structure_diffs)}")
    log.info(f"Particle data differences: {len(particle_diffs)}")

    if not structure_diffs and not particle_diffs:
        log.info(
            "✓ Files are structurally identical (only particle/tomogram counts differ)"
        )
    else:
        log.error("✗ Files have structural differences beyond particle/tomogram counts")
        raise ValueError(
            "Input and Output files have structural differences beyond particle/tomogram counts"
        )

    log.info("=" * 80)


def validate_mat_files(file1_path, file2_path):
    """Compare two .mat files."""
    log.info("=" * 80)
    log.info("MAT FILE COMPARISON")
    log.info("=" * 80)

    mat1 = load_mat_file(file1_path)
    mat2 = load_mat_file(file2_path)

    info1 = get_file_info(file1_path)
    info2 = get_file_info(file2_path)

    log_file_info(info1, 1)
    log_file_info(info2, 2)

    if not info1["exists"] or not info2["exists"]:
        raise FileNotFoundError("ERROR: One or both files do not exist!")

    log.info("Loading .mat files...")
    if mat1 is None or mat2 is None:
        raise ValueError("ERROR: Failed to load one or both .mat files!")

    log.info("Files loaded successfully!")

    log.info("Extracting geometry information...")
    geom_info1 = extract_geometry_info(mat1)
    geom_info2 = extract_geometry_info(mat2)

    log_geometry_info(geom_info1, 1)
    log_geometry_info(geom_info2, 2)

    structure_diffs = compare_mat_structures(mat1, mat2)
    particle_diffs = compare_particle_data_structure(mat1, mat2)

    log_comparison_results(structure_diffs, particle_diffs)
    log_summary(info1, info2, structure_diffs, particle_diffs)


def main():
    """Command line entry point."""
    # Set up logging when called from command line
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    if len(sys.argv) != 3:
        log.error("Usage: python test_mat_file_comparison.py <file1.mat> <file2.mat>")
        sys.exit(1)

    validate_mat_files(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
