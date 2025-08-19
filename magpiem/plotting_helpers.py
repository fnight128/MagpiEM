# -*- coding: utf-8 -*-
"""
Helper functions for plotting tomogram data without requiring Tomogram objects.
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Tuple, Optional
from .classes import simple_figure, colour_range


def create_particle_plot_from_raw_data(
    tomogram_raw_data: List[List[List[float]]],
    cone_size: float = 1.0,
    showing_removed_particles: bool = False,
    colour: str = "red",
    opacity: float = 1.0,
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

    Returns
    -------
    go.Figure
        Plotly figure with either particle positions (scatter3d) or cones
    """
    if not tomogram_raw_data:
        return go.Figure()

    positions = np.array([particle[0] for particle in tomogram_raw_data])
    orientations = np.array([particle[1] for particle in tomogram_raw_data])

    # Create base figure with standard formatting
    fig = simple_figure()

    if cone_size > 0:
        cone_trace = create_cone_traces(
            positions, orientations, cone_size, colour, opacity
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
        
    Returns
    -------
    go.Figure
        Plotly figure with separate traces for each lattice
    """
    if not tomogram_raw_data or not lattice_data:
        return simple_figure()
    
    # Create base figure
    fig = simple_figure()
    
    # Get the number of lattices and generate colours
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
            # Other lattices use the generated colour range
            # Use lattice_id to determine color index, ensuring consistent colors
            color_index = (lattice_id - 1) % len(lattice_colours)  # -1 because lattice 0 is handled separately
            colour = lattice_colours[color_index]
            opacity = 0.8
            
        # Create trace for this lattice
        if cone_size > 0:
            # Create cone trace for this lattice
            positions = np.array([p[0] for p in lattice_particles])
            orientations = np.array([p[1] for p in lattice_particles])
            
            cone_trace = create_cone_traces(
                positions, orientations, cone_size, colour, opacity
            )
            cone_trace.name = f"Lattice {lattice_id}"
            fig.add_trace(cone_trace)
        else:
            # Create scatter trace for this lattice
            positions = np.array([p[0] for p in lattice_particles])
            scatter_trace = create_scatter_trace(positions, colour, opacity)
            scatter_trace.name = f"Lattice {lattice_id}"
            fig.add_trace(scatter_trace)
    
    return fig


def create_scatter_trace(
    positions: np.ndarray, colour: str = "blue", opacity: float = 0.8
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
        showlegend=True,
    )


def generate_cone_fix_data(
    positions: np.ndarray, orientations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cone_fix data to ensure consistent cone sizing.
    See cone_fix_readme.txt for more details.

    Parameters
    ----------
    positions : np.ndarray
        Array of particle positions [N, 3]
    orientations : np.ndarray
        Array of particle orientations [N, 3]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (positions_with_fix, orientations_with_fix)
    """
    if len(positions) == 0:
        return positions, orientations

    # Calculate data range for cone_fix
    max_pos = np.max(positions, axis=0)
    min_pos = np.min(positions, axis=0)
    range_magnitude = np.linalg.norm(max_pos - min_pos)

    # Create cone_fix vectors (two tiny vectors close together)
    scaling = 1000000
    fudge_factor = 1000
    min_plus_mag = min_pos + range_magnitude / (scaling * fudge_factor)

    # Add cone_fix vectors to positions and orientations
    cone_fix_positions = np.vstack([positions, min_pos, min_plus_mag])
    cone_fix_orientations = np.vstack(
        [
            orientations,
            np.array(
                [1 / (3**0.5 * scaling)] * 3
            ),  # Tiny orientation for first fix vector
            np.array(
                [1 / (3**0.5 * scaling)] * 3
            ),  # Tiny orientation for second fix vector
        ]
    )

    return cone_fix_positions, cone_fix_orientations


def create_cone_traces(
    positions: np.ndarray,
    orientations: np.ndarray,
    cone_size: float,
    colour: str = "red",
    opacity: float = 1.0,
) -> go.Cone:
    """
    Create cone traces for particle orientations using Plotly's built-in Cone trace.
    Orientations are assumed to be normalised.

    Includes cone_fix mechanism to ensure consistent cone sizing.
    See cone_fix_readme.txt for more details.

    Parameters
    ----------
    positions : np.ndarray
        Array of particle positions [N, 3]
    orientations : np.ndarray
        Array of particle orientations [N, 3] (already normalised)
    cone_size : float
        Size of cones to plot
    colour : str, optional
        Colour for cones. Defaults to "red".
    opacity : float, optional
        Opacity for cones (0.0 to 1.0). Defaults to 1.0.

    Returns
    -------
    go.Cone
        Plotly cone trace
    """
    # Generate cone_fix data
    cone_fix_positions, cone_fix_orientations = generate_cone_fix_data(
        positions, orientations
    )

    return go.Cone(
        x=cone_fix_positions[:, 0],
        y=cone_fix_positions[:, 1],
        z=cone_fix_positions[:, 2],
        u=cone_fix_orientations[:, 0],
        v=cone_fix_orientations[:, 1],
        w=cone_fix_orientations[:, 2],
        sizemode="absolute",
        sizeref=cone_size,
        colorscale=[[0, colour], [1, colour]],
        showscale=False,
        opacity=opacity,
    )
