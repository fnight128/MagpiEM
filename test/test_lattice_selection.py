#!/usr/bin/env python3
"""
Test to verify the lattice selection functionality.
"""

import sys
import pathlib

# Add the parent directory to the path
current_dir = pathlib.Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from magpiem.read_write import read_emc_mat, read_emc_tomogram_raw_data
from magpiem.classes import Cleaner
from magpiem.dash_ui import clean_tomo_with_cpp
from magpiem.plotting_helpers import create_lattice_plot_from_raw_data


def test_lattice_selection():
    """Test that lattice selection functionality works."""
    print("Testing lattice selection functionality...")

    data = read_emc_mat("test/WT_CA_2nd.mat")

    # appropriate params for test data
    cleaner = Cleaner.from_user_params(2.0, 3, 10, 60.0, 40.0, 10.0, 20.0, 90.0, 20.0)

    # Test with wt2nd_4004_2
    tomo_name = "wt2nd_4004_2"
    raw_data = read_emc_tomogram_raw_data(data[tomo_name], tomo_name)

    # Run cleaning to get lattice data
    lattice_data = clean_tomo_with_cpp(raw_data, cleaner)

    print(f"Generated {len(lattice_data)} lattices for {tomo_name}")

    # Test plotting without selection
    fig_no_selection = create_lattice_plot_from_raw_data(
        raw_data, lattice_data, cone_size=0, show_removed_particles=False
    )
    print(f"Figure without selection has {len(fig_no_selection.data)} traces")

    # Test plotting with selection
    selected_lattices = {1, 3}  # Select lattices 1 and 3
    fig_with_selection = create_lattice_plot_from_raw_data(
        raw_data,
        lattice_data,
        cone_size=0,
        show_removed_particles=False,
        selected_lattices=selected_lattices,
    )
    print(f"Figure with selection has {len(fig_with_selection.data)} traces")

    # Check that selected lattices are plotted in white
    white_traces = 0
    for trace in fig_with_selection.data:
        if trace.marker.color == "white":
            white_traces += 1
            print(f"Found white trace: {trace.name}")

    print(f"Found {white_traces} white traces (should be 2)")

    # Save the figures for inspection
    fig_no_selection.write_html("test/lattice_plot_no_selection.html")
    fig_with_selection.write_html("test/lattice_plot_with_selection.html")

    print("Test completed. Check the HTML files to verify lattice selection works.")
    print("Selected lattices should appear in white, others in their normal colors.")


if __name__ == "__main__":
    test_lattice_selection()
