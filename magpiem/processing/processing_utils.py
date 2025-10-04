# -*- coding: utf-8 -*-
"""
Utility functions for processing operations in MagpiEM.
"""

import logging
from time import time
import datetime

logger = logging.getLogger(__name__)


def process_single_tomogram(
    tomo_name,
    full_geom,
    clean_params,
    set_progress,
    clean_count,
    total_tomos,
    clean_total_time,
    read_emc_tomogram_raw_data,
    clean_tomo_with_cpp,
    clean_and_detect_flips_with_cpp=None,
):
    """Process a single tomogram and return lattice data and timing info."""
    t0 = time()
    logger.debug(f"Cleaning tomogram: {tomo_name}")

    if tomo_name not in full_geom:
        logger.warning(f"Tomogram {tomo_name} not found in .mat file")
        return None, 0, None

    tomo_raw_data = read_emc_tomogram_raw_data(full_geom[tomo_name], tomo_name)
    if tomo_raw_data is None:
        logger.warning(f"Failed to extract data for tomogram: {tomo_name}")
        return None, 0, None

    # Use flip detection if allow_flips is True and function is provided
    if clean_params.allow_flips and clean_and_detect_flips_with_cpp is not None:
        lattice_data, flipped_particles = clean_and_detect_flips_with_cpp(
            tomo_raw_data, clean_params
        )
        flip_data = {tomo_name: flipped_particles}
    else:
        lattice_data = clean_tomo_with_cpp(tomo_raw_data, clean_params)
        flip_data = None

    processing_time = time() - t0

    progress_percent = int((clean_count / total_tomos) * 100)
    set_progress(progress_percent)

    # Log progress every 10 tomograms to reduce overhead
    if clean_count % 10 == 0 or clean_count == total_tomos:
        tomos_remaining = total_tomos - clean_count
        clean_speed = clean_total_time / clean_count
        secs_remaining = clean_speed * tomos_remaining
        formatted_time_remaining = str(
            datetime.timedelta(seconds=secs_remaining)
        ).split(".")[0]
        logger.info(
            f"Progress: {clean_count}/{total_tomos} ({progress_percent}%) - Time remaining: {formatted_time_remaining}"
        )

    return lattice_data, processing_time, flip_data
