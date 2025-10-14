#!/usr/bin/env python3
"""
Test for the find_flipped_particles method in the Tomogram class.

Loads flipped data, cleans, and finds flipped particles, plotting results.
"""

import sys
import numpy as np
from pathlib import Path

# Add test utilities to path
test_root = Path(__file__).parent.parent
sys.path.insert(0, str(test_root))

from test_utils import (  # noqa: E402
    TestConfig,
    setup_test_logging,
    get_test_data_path,
    log_test_start,
    log_test_success,
    log_test_failure,
    setup_test_environment,
)

setup_test_environment()

from magpiem.io.io_utils import read_single_tomogram  # noqa: E402
from magpiem.processing.classes.cleaner import Cleaner  # noqa: E402
from magpiem.processing.cpp_integration import (  # noqa: E402
    debug_flip_detection_with_cpp,
)
from magpiem.plotting.plotting_utils import (  # noqa: E402
    create_cone_traces,
)

logger = setup_test_logging()

TEST_DATA_FILE = get_test_data_path("test_data_flipped.mat")
ORIGINAL_DATA_FILE = get_test_data_path("test_data_single_lattice.mat")
TEST_TOMO_NAME = TestConfig.TEST_TOMO_STANDARD
TEST_CLEANER_VALUES = TestConfig.TEST_CLEANER_VALUES


def create_test_cleaner() -> Cleaner:
    """Create cleaning parameters optimized for small test dataset."""
    (
        cc_thresh,
        min_neigh,
        min_lattice_size,
        target_dist,
        dist_tol,
        target_ori,
        ori_tol,
        target_curv,
        curv_tol,
    ) = TEST_CLEANER_VALUES

    # More lenient parameters to ensure no issues with cleaning - not what we are testing here
    dist_range = Cleaner.dist_range(target_dist, dist_tol + 10.0)
    ori_range = Cleaner.ang_range_dotprod(target_ori, 60)
    curv_range = Cleaner.ang_range_dotprod(target_curv, 60)
    flipped_ori_range = tuple(-x for x in reversed(ori_range))

    return Cleaner(
        cc_thresh=cc_thresh,
        min_neighbours=0,
        min_lattice_size=1,
        dist_range=dist_range,
        ori_range=ori_range,
        curv_range=curv_range,
        allow_flips=True,
        flipped_ori_range=flipped_ori_range,
    )


def setup_test_tomogram():
    """Load and setup the test tomogram with cleaning parameters."""
    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError(f"Flipped test data not found: {TEST_DATA_FILE}")

    logger.info(f"Loading flipped test data from {TEST_DATA_FILE}")
    test_tomo = read_single_tomogram(str(TEST_DATA_FILE), TEST_TOMO_NAME)
    if test_tomo is None:
        raise ValueError(f"Failed to load tomogram: {TEST_TOMO_NAME}")

    logger.info(f"Loaded tomogram with {len(test_tomo.all_particles)} particles")

    # Create and assign cleaning parameters
    test_cleaner = create_test_cleaner()
    test_tomo.cleaning_params = test_cleaner

    logger.info(f"Assigned cleaning parameters: {test_cleaner}")
    logger.info(f"Allow flips: {test_cleaner.allow_flips}")

    # Run neighbour finding (required for flip detection)
    logger.info("Running neighbour finding...")
    test_tomo.find_particle_neighbours()
    logger.info("Neighbour finding completed")

    # Manually assign all particles to lattice 1
    for particle in test_tomo.all_particles:
        particle.set_lattice(1)
    test_tomo.lattices[1] = test_tomo.all_particles

    logger.info(
        f"Manually assigned {len(test_tomo.all_particles)} particles to lattice 1"
    )

    return test_tomo, test_cleaner


def find_actually_flipped_particles(test_tomo):
    """Compare with original dataset to identify actually flipped particles."""
    logger.info(
        "Comparing with original dataset to identify actually flipped particles..."
    )

    original_tomo = read_single_tomogram(str(ORIGINAL_DATA_FILE), TEST_TOMO_NAME)
    if not original_tomo:
        logger.info(
            "No actually flipped particles found - cannot calculate overlap percentage"
        )
        return [], []

    # Create particle lookup by ID
    orig_particles = {p.particle_id: p for p in original_tomo.all_particles}
    flip_particles = {p.particle_id: p for p in test_tomo.all_particles}

    actually_flipped = []
    for particle_id in orig_particles:
        if particle_id in flip_particles:
            orig_particle = orig_particles[particle_id]
            flip_particle = flip_particles[particle_id]

            # Use the dot_orientation method to compare orientations
            dot_prod = orig_particle.dot_orientation(flip_particle)
            if abs(dot_prod + 1) < 0.1:  # Close to -1 means 180° rotation
                actually_flipped.append(particle_id)

    logger.info(f"Found {len(actually_flipped)} actually flipped particles in dataset")
    logger.info(f"Actually flipped particle IDs: {actually_flipped}")

    return actually_flipped, original_tomo


def validate_flip_detection_results(
    detected_particle_ids, actually_flipped, implementation_name
):
    """
    Common validation function for flip detection results from any implementation.

    Parameters
    ----------
    detected_particle_ids : list
        List of particle IDs detected as flipped by the implementation
    actually_flipped : list
        List of particle IDs that are actually flipped (ground truth)
    implementation_name : str
        Name of the implementation being validated (for logging)
    """
    # Check overlap with our detected flipped particles
    overlap = set(actually_flipped) & set(detected_particle_ids)
    logger.info(
        f"{implementation_name} overlap between detected and truly flipped "
        f"particles: {len(overlap)} particles"
    )

    assert (
        len(actually_flipped) > 0
    ), "No flipped particles found from original dataset, test is invalid"
    overlap_percentage = len(overlap) / len(actually_flipped) * 100
    logger.info(f"{implementation_name} overlap percentage: {overlap_percentage:.1f}%")

    # Test should fail if overlap isn't perfect
    assert len(overlap) == len(actually_flipped), (
        f"{implementation_name} expected perfect overlap but got "
        f"{len(overlap)}/{len(actually_flipped)} particles"
    )
    assert set(detected_particle_ids) == set(actually_flipped), (
        f"{implementation_name} detected particles {set(detected_particle_ids)} "
        f"don't match actually flipped particles {set(actually_flipped)}"
    )


def run_python_flip_detection(test_tomo):
    """Run the Python flip detection implementation and return results."""
    logger.info("Running Python flip detection...")

    # Run Python flip detection
    flipped_particles = test_tomo.find_flipped_particles()
    logger.info(f"Python found {len(flipped_particles)} flipped particles")

    # Convert to particle IDs
    python_flipped_particle_ids = [p.particle_id for p in flipped_particles]

    return flipped_particles, python_flipped_particle_ids


def run_cpp_flip_detection(test_tomo, test_cleaner):
    """Run the C++ debug implementation and return results."""
    logger.info("Running C++ debug implementation...")

    # Prepare raw data for C++ function
    raw_data = []
    for particle in test_tomo.all_particles:
        raw_data.append([particle.position, particle.orientation])

    # Run C++ debug implementation
    debug_lattice_data, debug_flipped_indices = debug_flip_detection_with_cpp(
        raw_data, test_cleaner
    )

    logger.info(f"Debug C++ found {len(debug_flipped_indices)} flipped particles")

    # Convert C++ indices to particle IDs
    all_particles_list = list(test_tomo.all_particles)
    debug_flipped_particle_ids = []
    for idx in debug_flipped_indices:
        debug_flipped_particle_ids.append(all_particles_list[idx].particle_id)

    return debug_flipped_indices, debug_flipped_particle_ids


def compare_python_cpp_results(python_flipped_ids, debug_flipped_particle_ids):
    """Compare Python and C++ results to ensure they match."""
    # Python and C++ debug should give identical results
    assert set(python_flipped_ids) == set(debug_flipped_particle_ids), (
        f"Python and Debug C++ results don't match: "
        f"Python={set(python_flipped_ids)}, Debug C++={set(debug_flipped_particle_ids)}"
    )


def create_visualization(test_tomo, flipped_particles, debug_flipped_indices):
    """Create and save the visualization plot."""
    logger.info("Visualising...")

    # Create the base lattice plot
    fig = test_tomo.plot_all_lattices(
        cone_size=10.0,
        cone_fix=True,
        showing_removed_particles=False,
    )

    # Add Python flipped particles in red
    if flipped_particles:
        python_flipped_positions = []
        python_flipped_orientations = []

        for particle in flipped_particles:
            python_flipped_positions.append(particle.position)
            python_flipped_orientations.append(particle.orientation)

        if python_flipped_positions:
            python_flipped_positions = np.array(python_flipped_positions)
            python_flipped_orientations = np.array(python_flipped_orientations)

            python_flipped_trace = create_cone_traces(
                positions=python_flipped_positions,
                orientations=python_flipped_orientations,
                cone_size=0.8,
                colour="red",
                opacity=0.5,
                lattice_id=-1,  # Dummy ID for flipped particles
            )
            python_flipped_trace.name = "Python Flipped Particles"
            fig.add_trace(python_flipped_trace)

    # Add Debug C++ flipped particles in blue
    if debug_flipped_indices:
        debug_flipped_positions = []
        debug_flipped_orientations = []
        all_particles_list = list(test_tomo.all_particles)

        for idx in debug_flipped_indices:
            particle = all_particles_list[idx]
            debug_flipped_positions.append(particle.position)
            debug_flipped_orientations.append(particle.orientation)

        if debug_flipped_positions:
            debug_flipped_positions = np.array(debug_flipped_positions)
            debug_flipped_orientations = np.array(debug_flipped_orientations)

            debug_flipped_trace = create_cone_traces(
                positions=debug_flipped_positions,
                orientations=debug_flipped_orientations,
                cone_size=0.8,
                colour="blue",
                opacity=0.5,
                lattice_id=-2,  # Dummy ID for Debug C++ flipped particles
            )
            debug_flipped_trace.name = "Debug C++ Flipped Particles"
            fig.add_trace(debug_flipped_trace)

    # Save the plot
    output_html = test_root / "logs" / "find_flipped_particles_result.html"
    logger.info(f"Saving interactive plot to {output_html}")
    fig.write_html(str(output_html))

    return fig, output_html


def validate_test_results(
    fig,
    output_html,
    flipped_particles,
    total_particles,
    debug_flipped_indices,
    python_flipped_ids,
    debug_flipped_particle_ids,
):
    """Validate final test results and log summary."""
    # Basic assertions
    assert fig is not None, "Figure was not created"
    assert output_html.exists(), "Output HTML file was not created"
    assert isinstance(
        flipped_particles, list
    ), f"Flipped particles returned as an incorrect type, {type(flipped_particles)}"

    flipped_count = len(flipped_particles)
    flipped_percentage = (
        (flipped_count / total_particles * 100) if total_particles > 0 else 0
    )

    logger.info("Test completed successfully")
    logger.info(f"Total particles: {total_particles}")
    logger.info(
        f"Python flipped particles: {flipped_count} ({flipped_percentage:.1f}%)"
    )
    logger.info(
        f"Debug C++ flipped particles: {len(debug_flipped_indices)} "
        f"({len(debug_flipped_indices) / total_particles * 100:.1f}%)"
    )
    logger.info(
        f"Python and Debug C++ results match: "
        f"{set(python_flipped_ids) == set(debug_flipped_particle_ids)}"
    )
    logger.info(f"Interactive plot saved as: {output_html}")


def test_find_flipped_particles():
    """Test the find_flipped_particles method with visualisation."""
    test_name = "Find Flipped Particles Test"
    log_test_start(test_name, logger)

    try:
        # Setup test data
        test_tomo, test_cleaner = setup_test_tomogram()

        # Find actually flipped particles from original dataset
        actually_flipped, original_tomo = find_actually_flipped_particles(test_tomo)

        # Run Python flip detection
        flipped_particles, python_flipped_particle_ids = run_python_flip_detection(
            test_tomo
        )

        # Debug: Check which particles are being identified as flipped
        logger.info(
            f"Python flipped particle IDs (first 10): {python_flipped_particle_ids[:10]}"
        )
        for particle in flipped_particles[:5]:  # Log first 5 for debugging
            logger.debug(
                f"Flipped particle ID: {particle.particle_id}, "
                f"Lattice: {particle.lattice}, "
                f"Direction: {getattr(particle, 'direction', 'None')}"
            )

        # Run C++ flip detection
        debug_flipped_indices, debug_flipped_particle_ids = run_cpp_flip_detection(
            test_tomo, test_cleaner
        )

        # Create visualization
        fig, output_html = create_visualization(
            test_tomo, flipped_particles, debug_flipped_indices
        )

        # Validate both implementations using common validation function
        if actually_flipped:
            validate_flip_detection_results(
                python_flipped_particle_ids, actually_flipped, "Python"
            )
            validate_flip_detection_results(
                debug_flipped_particle_ids, actually_flipped, "C++"
            )

        # Compare Python and C++ results
        compare_python_cpp_results(
            python_flipped_particle_ids, debug_flipped_particle_ids
        )

        # Validate final results
        validate_test_results(
            fig,
            output_html,
            flipped_particles,
            len(test_tomo.all_particles),
            debug_flipped_indices,
            python_flipped_particle_ids,
            debug_flipped_particle_ids,
        )

        log_test_success(test_name, logger)

    except Exception as e:
        log_test_failure(test_name, e, logger)
        raise


if __name__ == "__main__":
    test_find_flipped_particles()
    print("Test completed successfully!")
