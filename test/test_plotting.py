#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the new plotting system.
Loads test_data.mat and plots particles in the wt2nd_4004_2 tomogram.
"""

import sys
import os
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Add the parent directory to the path so we can import magpiem
sys.path.insert(0, str(Path(__file__).parent.parent))

from magpiem.read_write import read_multiple_tomograms_raw_data
from magpiem.plotting_helpers import create_particle_plot_from_raw_data

TEST_DATA_FILENAME = "test_data.mat"
TEST_TOMOGRAM_NAME = "wt2nd_4004_2"


def test_plotting():
    """Test the new plotting system with test data."""

    # Path to test data
    test_file = Path(__file__).parent / TEST_DATA_FILENAME

    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        return

    print(f"Loading test data from: {test_file}")

    # Load raw tomogram data
    tomogram_data = read_multiple_tomograms_raw_data(str(test_file))

    if tomogram_data is None:
        print("Error: Could not load tomogram data")
        return

    print(f"Loaded {len(tomogram_data)} tomograms:")
    for tomo_name in tomogram_data.keys():
        print(f"  - {tomo_name}")

    # Check if our target tomogram exists
    if TEST_TOMOGRAM_NAME not in tomogram_data:
        print(f"Error: Tomogram '{TEST_TOMOGRAM_NAME}' not found in data")
        print(f"Available tomograms: {list(tomogram_data.keys())}")
        return

    # Get the raw data for our target tomogram
    raw_data = tomogram_data[TEST_TOMOGRAM_NAME]
    print(f"\nTomogram '{TEST_TOMOGRAM_NAME}' contains {len(raw_data)} particles")

    # 1. Scatter plot (cone_size <= 0)
    print("1. Creating scatter plot...")
    scatter_fig = create_particle_plot_from_raw_data(
        raw_data,
        cone_size=-1,  # Negative value for scatter plot
        showing_removed_particles=False,
    )

    # Save scatter plot
    scatter_output = Path(__file__).parent / f"{TEST_TOMOGRAM_NAME}_scatter.html"
    scatter_fig.write_html(str(scatter_output))
    print(f"   Saved scatter plot to: {scatter_output}")

    # 2. Cone plot (cone_size > 0)
    print("2. Creating cone plot...")
    cone_fig = create_particle_plot_from_raw_data(
        raw_data,
        cone_size=10.0,  # Positive value for cone plot
        showing_removed_particles=False,
    )

    # Save cone plot
    cone_output = Path(__file__).parent / f"{TEST_TOMOGRAM_NAME}_cones.html"
    cone_fig.write_html(str(cone_output))
    print(f"   Saved cone plot to: {cone_output}")

    # 3. Show some statistics about the data
    print("\nData statistics:")
    positions = np.array([particle[0] for particle in raw_data])
    orientations = np.array([particle[1] for particle in raw_data])

    print(f"  Position range:")
    print(f"    X: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f}")
    print(f"    Y: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f}")
    print(f"    Z: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f}")

    # Check if orientations are normalised for the sake of testing
    norms = np.linalg.norm(orientations, axis=1)
    print(
        f"  Orientation norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}"
    )

    if np.allclose(norms, 1.0, atol=1e-6):
        print("  ✓ Orientations are properly normalised")
    else:
        print("  ⚠ Orientations may not be normalised")

    print(f"\nTest completed successfully!")
    print(f"Open the HTML files in your browser to view the plots:")
    print(f"  - {scatter_output}")
    print(f"  - {cone_output}")


if __name__ == "__main__":
    test_plotting()
