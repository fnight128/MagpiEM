# -*- coding: utf-8 -*-
"""
Helper functions for plotting tomogram data without requiring Tomogram objects.
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import colorsys
from typing import List, Dict, Tuple, Optional


def simple_figure() -> go.Figure:
    """
    Returns a simple empty figure with generally appropriate settings for particle display
    """
    layout = go.Layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig = go.Figure(layout=layout)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(scene_aspectmode="data")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10})
    fig.update_layout(showlegend=False)
    fig["layout"]["uirevision"] = "a"
    return fig


def colour_range(num_points: int) -> list[str]:
    """
    Create an even range of colours across the spectrum

    Parameters
    ----------
    num_points
        Number of colours to create

    Returns
    -------
        List of colours in the form "rgb({},{},{})"
    """
    hsv_tuples = [(x * 1.0 / num_points, 0.75, 0.75) for x in range(num_points)]
    rgb_tuples = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    return [
        "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
        for (r, g, b) in rgb_tuples
    ]


def create_particle_plot_from_raw_data(
    tomogram_raw_data: List[List[List[float]]],
    cone_size: float = 1.0,
    showing_removed_particles: bool = False,
    colour: str = "red",
    opacity: float = 1.0,
    flipped_particle_indices: list = None,
) -> go.Figure:
    """
    Create a particle plot from raw tomogram data without requiring a Tomogram object.
    Particle orientations are assumed to be normalised.

    Parameters
    ----------
    tomogram_raw_data : List[List[List[float]]]
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]
    cone_size : float, optional
        Size of cones to plot. If <=0, scatter points are shown instead.
        Defaults to 1.0.
    showing_removed_particles : bool, optional
        Whether to show removed particles.
        Defaults to False.
    colour : str, optional
        Colour for particles/cones. Defaults to "red".
    opacity : float, optional
        Opacity for particles/cones (0.0 to 1.0). Defaults to 1.0.
    flipped_particle_indices : list, optional
        List of particle indices that should have their orientations flipped.
        If None, no particles are flipped. Defaults to None.

    Returns
    -------
    go.Figure
        Plotly figure with either particle positions (scatter3d) or cones
    """
    if not tomogram_raw_data:
        return simple_figure()

    positions = np.array([particle[0] for particle in tomogram_raw_data])
    orientations = np.array([particle[1] for particle in tomogram_raw_data])

    # Flip orientations if requested
    if flipped_particle_indices:
        for idx in flipped_particle_indices:
            if idx < len(orientations):
                # for plotting purposes, invert orientation with a simple reflection
                # note that this is NOT equivalent to the 180 degree rotation
                # used for fixing the actual saved data (as this is not a proper rotation)
                orientations[idx] = -orientations[idx]

    # Create base figure with standard formatting
    fig = simple_figure()

    if cone_size > 0:
        # Generate cone fix points for this plot
        cone_fix_positions, cone_fix_orientations = generate_cone_fix_points(
            tomogram_raw_data
        )

        # Append cone fix points to positions and orientations
        positions_with_fix, orientations_with_fix = append_cone_fix_to_lattice(
            positions, orientations, cone_fix_positions, cone_fix_orientations
        )

        cone_trace = create_cone_traces(
            positions_with_fix, orientations_with_fix, cone_size, colour, opacity
        )
        fig.add_trace(cone_trace)
    else:
        scatter_trace = create_scatter_trace(positions, colour, opacity)
        fig.add_trace(scatter_trace)

    return fig


def create_lattice_plot_from_raw_data(
    tomogram_raw_data: List[List[List[float]]],
    lattice_data: Dict[int, List[int]],
    cone_size: float = 1.0,
    show_removed_particles: bool = False,
    selected_lattices: Optional[set] = {},
    flipped_particle_indices: list = None,
) -> go.Figure:
    """
    Create a particle plot with separate traces for each lattice.

    Parameters
    ----------
    tomogram_raw_data : List[List[List[float]]]
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]
    lattice_data : Dict[int, List[int]]
        Dictionary mapping lattice_id -> list of particle indices
    cone_size : float, optional
        Size of cones to plot. If <=0, scatter points are shown instead.
        Defaults to 1.0.
    show_removed_particles : bool, optional
        Whether to show removed particles (lattice 0). Defaults to False.
    selected_lattices : Optional[set], optional
        Set of lattice IDs that are selected. Selected lattices will be plotted in white.
        Defaults to None.
    flipped_particle_indices : list, optional
        List of particle indices that should have their orientations flipped.
        If None, no particles are flipped. Defaults to None.

    Returns
    -------
    go.Figure
        Plotly figure with separate traces for each lattice
    """
    if not tomogram_raw_data or not lattice_data:
        return simple_figure()

    fig = simple_figure()

    cone_fix_positions, cone_fix_orientations = generate_cone_fix_points(
        tomogram_raw_data
    )

    lattice_ids = list(lattice_data.keys())
    num_lattices = len(lattice_ids)
    lattice_colours = colour_range(num_lattices)

    for lattice_id, particle_ids in lattice_data.items():
        # Convert lattice_id to int in case it was serialized as string by dcc.Store
        lattice_id = int(lattice_id)
        if len(particle_ids) == 0:
            continue

        # Skip lattice 0 (removed particles) unless show_removed_particles is True
        if lattice_id == 0 and not show_removed_particles:
            continue

        # Extract particles for this lattice
        lattice_particles = [tomogram_raw_data[j] for j in particle_ids]

        # Choose colour and opacity for this lattice
        if lattice_id == 0:
            # Lattice 0 (removed particles) should be black with opacity 0.6
            colour = "black"
            opacity = 0.6
        else:
            # Check if this lattice is selected
            if selected_lattices and lattice_id in selected_lattices:
                # Selected lattices are plotted in white
                colour = "white"
                opacity = 0.9
            else:
                # Other lattices use the generated colour range
                # Use lattice_id to determine color index, ensuring consistent colors
                color_index = (lattice_id - 1) % len(
                    lattice_colours
                )  # -1 because lattice 0 is handled separately
                colour = lattice_colours[color_index]
                opacity = 0.8

        # Create trace
        if cone_size > 0:
            positions = np.array([p[0] for p in lattice_particles])
            orientations = np.array([p[1] for p in lattice_particles])

            # Flip orientations for flipped particles if requested
            if flipped_particle_indices:
                for i, particle_idx in enumerate(particle_ids):
                    if particle_idx in flipped_particle_indices:
                        orientations[i] = -orientations[i]

            positions_with_fix, orientations_with_fix = append_cone_fix_to_lattice(
                positions, orientations, cone_fix_positions, cone_fix_orientations
            )

            cone_trace = create_cone_traces(
                positions_with_fix,
                orientations_with_fix,
                cone_size,
                colour,
                opacity,
                lattice_id,
            )
            cone_trace.name = f"Lattice {lattice_id}"
            fig.add_trace(cone_trace)
        else:
            # Create scatter trace for this lattice
            positions = np.array([p[0] for p in lattice_particles])

            scatter_trace = create_scatter_trace(positions, colour, opacity, lattice_id)
            scatter_trace.name = f"Lattice {lattice_id}"
            fig.add_trace(scatter_trace)

    return fig


def update_lattice_trace_colors(
    fig: go.Figure,
    selected_lattices: set,
    lattice_data: Dict[int, List[int]],
    cone_size: float = 1.0,
) -> go.Figure:
    """
    Update trace colors for selected lattices without recreating the entire plot.
    This is much more efficient for large datasets.

    Parameters
    ----------
    fig : go.Figure
        Existing plotly figure to update
    selected_lattices : set
        Set of lattice IDs that are selected
    lattice_data : Dict[int, List[int]]
        Dictionary mapping lattice_id -> list of particle indices
    cone_size : float, optional
        Size of cones to plot. If <=0, scatter points are shown instead.
        Defaults to 1.0.

    Returns
    -------
    go.Figure
        Updated figure with modified trace colors
    """
    if not fig.data:
        return fig

    # Get the number of lattices and generate colours
    lattice_ids = list(lattice_data.keys())
    num_lattices = len(lattice_ids)
    lattice_colours = colour_range(num_lattices)

    # Update each trace based on selection state
    for trace in fig.data:
        # Extract lattice ID from trace name
        if not trace.name or not trace.name.startswith("Lattice "):
            continue

        try:
            lattice_id = int(trace.name.split(" ")[1])
        except (ValueError, IndexError):
            continue

        # Skip lattice 0 (removed particles)
        if lattice_id == 0:
            continue

        # Determine new color and opacity
        if lattice_id in selected_lattices:
            # Selected lattices are plotted in white
            new_color = "white"
            new_opacity = 0.9
        else:
            # Other lattices use the generated colour range
            color_index = (lattice_id - 1) % len(lattice_colours)
            new_color = lattice_colours[color_index]
            new_opacity = 0.8

        # Update trace color and opacity
        if hasattr(trace, "marker"):
            # Scatter trace
            trace.marker.color = new_color
            trace.marker.opacity = new_opacity
        elif hasattr(trace, "colorscale"):
            # Cone trace - update colorscale and opacity
            trace.colorscale = [[0, new_color], [1, new_color]]
            trace.opacity = new_opacity
        else:
            # Fallback for other trace types
            trace.opacity = new_opacity

    return fig


def add_selected_points_trace(
    fig: go.Figure,
    clicked_point_data: list,
    trace_name: str = "selected_particle_trace",
) -> go.Figure:
    """
    Add or update selected points trace without recreating the entire plot.

    Parameters
    ----------
    fig : go.Figure
        Existing plotly figure to update
    clicked_point_data : list
        List of point data dictionaries, each containing x, y, z coordinates
    trace_name : str, optional
        Name for the selected points trace. Defaults to "selected_particle_trace".

    Returns
    -------
    go.Figure
        Updated figure with selected points trace
    """
    if not clicked_point_data:
        return fig

    # Extract point coordinates
    selected_points = []
    for point_data in clicked_point_data:
        selected_points.append([point_data["x"], point_data["y"], point_data["z"]])

    if not selected_points:
        return fig

    # Remove existing selected points trace if it exists
    fig.data = [trace for trace in fig.data if trace.name != trace_name]

    # Add new selected points trace
    positions = np.array(selected_points)
    particles_scatter_trace = create_scatter_trace(
        positions, colour="black", opacity=0.8
    )
    particles_scatter_trace.name = trace_name
    particles_scatter_trace.marker.size = 8  # Override default size for selected points
    particles_scatter_trace.showlegend = False  # Ensure legend is hidden
    fig.add_trace(particles_scatter_trace)

    return fig


def remove_selected_points_trace(
    fig: go.Figure, trace_name: str = "selected_particle_trace"
) -> go.Figure:
    """
    Remove selected points trace from the figure.

    Parameters
    ----------
    fig : go.Figure
        Existing plotly figure to update
    trace_name : str, optional
        Name of the selected points trace to remove. Defaults to "selected_particle_trace".

    Returns
    -------
    go.Figure
        Updated figure without selected points trace
    """
    fig.data = [trace for trace in fig.data if trace.name != trace_name]
    return fig


def generate_constant_labels(num_particles: int, label: str) -> List:
    """
    Generate a list of constant labels for a given number of particles.
    """
    return [label] * num_particles


def create_scatter_trace(
    positions: np.ndarray,
    colour: str = "blue",
    opacity: float = 0.8,
    lattice_id: int = 0,
) -> go.Scatter3d:
    """
    Create scatter trace for particle positions.

    Parameters
    ----------
    positions : np.ndarray
        Array of particle positions [N, 3]
    colour : str, optional
        Colour for scatter points. Defaults to "blue".
    opacity : float, optional
        Opacity for scatter points (0.0 to 1.0). Defaults to 0.8.
    lattice_id : int, optional
        Lattice ID to use for text labels. Defaults to 0.

    Returns
    -------
    go.Scatter3d
        Scatter trace
    """
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker=dict(size=3, color=colour, opacity=opacity),
        name="Particles",
        showlegend=False,
        text=generate_constant_labels(len(positions), lattice_id),
    )


def generate_cone_fix_points(
    tomogram_raw_data: List[List[List[float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the two cone fix points that should be added to all lattices.
    These points ensure consistent cone sizing across all lattices in a plot.

    Parameters
    ----------
    tomogram_raw_data : List[List[List[float]]]
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (cone_fix_positions, cone_fix_orientations) - just the two fix points
    """
    if not tomogram_raw_data:
        return np.array([]), np.array([])

    positions = np.array([particle[0] for particle in tomogram_raw_data])

    # Calculate data range for cone fix points
    max_pos = np.max(positions, axis=0)
    min_pos = np.min(positions, axis=0)
    range_magnitude = np.linalg.norm(max_pos - min_pos)

    # Create cone fix vectors (two tiny vectors close together)
    scaling = 1000000
    fudge_factor = 1000
    min_plus_mag = min_pos + range_magnitude / (scaling * fudge_factor)

    cone_fix_positions = np.vstack([min_pos, min_plus_mag])
    cone_fix_orientations = np.vstack(
        [
            np.array(
                [1 / (3**0.5 * scaling)] * 3
            ),  # Tiny orientation for first fix vector
            np.array(
                [1 / (3**0.5 * scaling)] * 3
            ),  # Tiny orientation for second fix vector
        ]
    )

    return cone_fix_positions, cone_fix_orientations


def append_cone_fix_to_lattice(
    positions: np.ndarray,
    orientations: np.ndarray,
    cone_fix_positions: np.ndarray,
    cone_fix_orientations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Append cone fix points to a lattice's positions and orientations.

    Parameters
    ----------
    positions : np.ndarray
        Array of particle positions [N, 3]
    orientations : np.ndarray
        Array of particle orientations [N, 3]
    cone_fix_positions : np.ndarray
        Array of cone fix positions [2, 3]
    cone_fix_orientations : np.ndarray
        Array of cone fix orientations [2, 3]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (positions_with_fix, orientations_with_fix)
    """
    if len(positions) == 0:
        return positions, orientations

    # Append cone fix points to positions and orientations
    positions_with_fix = np.vstack([positions, cone_fix_positions])
    orientations_with_fix = np.vstack([orientations, cone_fix_orientations])

    return positions_with_fix, orientations_with_fix


def create_cone_traces(
    positions: np.ndarray,
    orientations: np.ndarray,
    cone_size: float,
    colour: str = "red",
    opacity: float = 1.0,
    lattice_id: int = 0,
) -> go.Cone:
    """
    Create cone traces for particle orientations using Plotly's built-in Cone trace.
    Orientations are assumed to be normalised.

    Parameters
    ----------
    positions : np.ndarray
        Array of particle positions [N, 3] (should include cone fix points if needed)
    orientations : np.ndarray
        Array of particle orientations [N, 3] (should be normalised, and should include cone fix orientations if needed)
    cone_size : float
        Size of cones to plot
    colour : str, optional
        Colour for cones. Defaults to "red".
    opacity : float, optional
        Opacity for cones (0.0 to 1.0). Defaults to 1.0.
    lattice_id : int, optional
        Lattice ID to use for text labels. Defaults to 0.

    Returns
    -------
    go.Cone
        Plotly cone trace
    """
    return go.Cone(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        u=orientations[:, 0],
        v=orientations[:, 1],
        w=orientations[:, 2],
        sizemode="absolute",
        sizeref=cone_size,
        colorscale=[[0, colour], [1, colour]],
        showscale=False,
        showlegend=False,
        opacity=opacity,
        text=generate_constant_labels(len(positions), lattice_id),
    )


def get_tomogram_figure(
    selected_tomo_name,
    tomogram_raw_data,
    session_key,
    lattice_data,
    make_cones,
    cone_size,
    show_removed,
    flip_data,
    get_cached_tomogram_figure,
    EMPTY_FIG,
):
    """Get cached tomogram figure with proper validation."""
    import logging

    logger = logging.getLogger(__name__)

    if not selected_tomo_name or not tomogram_raw_data:
        logger.debug("Returning EMPTY_FIG - missing data")
        return EMPTY_FIG

    if selected_tomo_name not in tomogram_raw_data["__tomogram_names__"]:
        logger.debug("Returning EMPTY_FIG - tomogram not found")
        return EMPTY_FIG

    data_path = tomogram_raw_data["__data_path__"]
    logger.debug("Getting cached figure for %s", selected_tomo_name)

    actual_cone_size = cone_size if make_cones else -1
    logger.debug("actual_cone_size=%s", actual_cone_size)

    # Get flipped particle indices for this tomogram
    flipped_particle_indices = (
        flip_data.get(selected_tomo_name, []) if flip_data else []
    )

    fig = get_cached_tomogram_figure(
        selected_tomo_name,
        data_path,
        session_key,
        lattice_data,
        actual_cone_size,
        show_removed,
        flipped_particle_indices,
    )

    logger.debug("Got figure from cache: %s", fig is not None)
    return fig if fig is not None else EMPTY_FIG


def apply_figure_customizations(
    fig,
    selected_lattices,
    selected_tomo_name,
    lattice_data,
    make_cones,
    cone_size,
    clicked_point_data,
    camera_data,
    EMPTY_FIG,
    TEMP_TRACE_NAME,
):
    """Apply lattice selection, point selection, and camera position to figure."""
    import logging

    logger = logging.getLogger(__name__)

    if fig is None or fig == EMPTY_FIG:
        return fig

    # Update lattice selection by trace update
    if selected_lattices and selected_tomo_name in selected_lattices:
        logger.debug(
            "Updating lattice colors for %s selected lattices",
            len(selected_lattices[selected_tomo_name]),
        )
        fig = update_lattice_trace_colors(
            fig,
            set(selected_lattices[selected_tomo_name]),
            lattice_data.get(selected_tomo_name, {}),
            cone_size if make_cones else -1,
        )

    # Add selected points if any
    if clicked_point_data:
        logger.debug("Adding %s selected points", len(clicked_point_data))
        fig = add_selected_points_trace(fig, clicked_point_data, TEMP_TRACE_NAME)

    # Restore camera position
    if camera_data:
        fig["layout"]["scene"]["camera"] = camera_data

    return fig


def extract_point_data(click_data, POSITION_KEYS, ORIENTATION_KEYS):
    """Extract particle data from click data."""
    current_point_data = click_data["points"][0]
    particle_data_keys = POSITION_KEYS + ORIENTATION_KEYS
    return {key: current_point_data[key] for key in particle_data_keys}


def calculate_geometric_params(clicked_points_data, particle_from_point_data, html):
    """Calculate geometric parameters between two selected points."""
    selected_particles = []
    for idx, point_data in enumerate(clicked_points_data):
        selected_particles.append(particle_from_point_data(point_data, idx=idx))

    # Check if points are too close together
    if selected_particles[0].distance_sq(selected_particles[1]) < 0.001:
        return None, "Points too close together"

    params_dict = selected_particles[0].calculate_params(selected_particles[1])
    params_message = []
    for param_name, param_value in params_dict.items():
        params_message.append(f"{param_name}: {param_value:.2f}")
        params_message.append(html.Br())

    return params_message, None
