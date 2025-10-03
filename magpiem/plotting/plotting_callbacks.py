# -*- coding: utf-8 -*-
"""
Plotting and visualization callbacks for MagpiEM.
"""

import logging
from dash import State, ctx, dcc, html
from dash_extensions.enrich import Input, Output

from ..processing.classes.particle import Particle
from ..processing.classes.tomogram import Tomogram
from ..dash.layout import EMPTY_FIG
from .plotting_utils import (
    get_tomogram_figure,
    apply_figure_customizations,
    extract_point_data,
    calculate_geometric_params,
)
from .plot_cache import clear_cache

logger = logging.getLogger(__name__)

# Constants
CAMERA_KEY = "scene.camera"
POSITION_KEYS = ["x", "y", "z"]
ORIENTATION_KEYS = ["u", "v", "w"]
TEMP_TRACE_NAME = "selected_particle_trace"


def register_plotting_callbacks(app, cache_functions, temp_file_dir):
    """Register plotting and visualization callbacks."""

    # Unpack cache functions
    get_cached_tomogram_figure = cache_functions["get_cached_tomogram_figure"]
    preload_tomograms = cache_functions["preload_tomograms"]

    def particle_from_point_data(point_data: dict, idx=0) -> Particle:
        return Particle.from_dict(
            {
                "particle_id": idx,
                "cc_score": 1,
                "position": [point_data[key] for key in POSITION_KEYS],
                "orientation": [point_data[key] for key in ORIENTATION_KEYS],
                "lattice": 1,
            },
            Tomogram(""),
        )

    @app.callback(
        Output("store-cache-cleared", "data"),
        Input("switch-cone-plot", "on"),
        Input("inp-cone-size", "value"),
        Input("button-set-cone-size", "n_clicks"),
        Input("switch-show-removed", "on"),
        Input("switch-flip-particles", "on"),
        State("store-session-key", "data"),
        State("store-tomogram-data", "data"),
        State("store-lattice-data", "data"),
        State("store-selected-lattices", "data"),
        State("dropdown-tomo", "value"),
        State("store-flips", "data"),
        prevent_initial_call=True,
    )
    def handle_plot_style_changes(
        make_cones,
        cone_size,
        cone_clicks,
        show_removed,
        flip_particles,
        session_key,
        tomogram_raw_data,
        lattice_data,
        selected_lattices,
        selected_tomo_name,
        flip_data,
    ):
        """Handle plot style changes and update cache accordingly."""
        logger.debug("handle_plot_style_changes called")
        logger.debug(
            "make_cones=%s, cone_size=%s, show_removed=%s, flip_particles=%s",
            make_cones,
            cone_size,
            show_removed,
            flip_particles,
        )

        if not session_key or not tomogram_raw_data or not selected_tomo_name:
            logger.debug("Missing required data, returning True")
            return True

        if not make_cones:
            cone_size = -1
            logger.debug("Cones disabled, setting cone_size to -1")

        # Clear cache and create new preloaded figures with updated style
        logger.debug("Clearing cache")
        clear_cache(session_key)
        try:
            data_path = tomogram_raw_data["__data_path__"]
            logger.debug("Preloading tomograms with new style")
            preload_tomograms(
                selected_tomo_name,
                tomogram_raw_data["__tomogram_names__"],
                data_path,
                session_key,
                lattice_data,
                cone_size,
                show_removed,
                flip_data,
            )
        except Exception as e:
            logger.debug("Failed to update cache: %s", e)
            logger.warning(f"Failed to update cache with new plot style: {e}")

        logger.debug("Returning True to trigger cache clear")
        return True

    @app.callback(
        Output("graph-picking", "figure"),
        Input("dropdown-tomo", "value"),
        Input("store-lattice-data", "data"),
        Input("store-selected-lattices", "data"),
        Input("store-clicked-point", "data"),
        Input("store-cache-cleared", "data"),
        State("store-tomogram-data", "data"),
        State("store-camera", "data"),
        State("store-session-key", "data"),
        State("switch-cone-plot", "on"),
        State("inp-cone-size", "value"),
        State("switch-show-removed", "on"),
        State("switch-flip-particles", "on"),
        State("store-flips", "data"),
        prevent_initial_call=True,
    )
    def plot_tomo(
        selected_tomo_name,
        lattice_data,
        selected_lattices,
        clicked_point_data,
        cache_cleared,
        tomogram_raw_data,
        camera_data,
        session_key,
        make_cones,
        cone_size,
        show_removed,
        flip_particles,
        flip_data,
    ):
        """Retrieve and plot cached tomo"""
        logger.debug("plot_tomo called with selected_tomo_name=%s", selected_tomo_name)
        logger.debug("make_cones=%s, cone_size=%s", make_cones, cone_size)
        logger.debug("cache_cleared=%s", cache_cleared)

        fig = get_tomogram_figure(
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
        )

        fig = apply_figure_customizations(
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
        )

        logger.debug("Returning figure")
        return fig

    @app.callback(
        Output("store-selected-lattices", "data"),
        Input("graph-picking", "clickData"),
        State("store-selected-lattices", "data"),
        State("store-lattice-data", "data"),
        State("dropdown-tomo", "value"),
        prevent_initial_call=True,
    )
    def handle_lattice_selection(
        click_data, selected_lattices, lattice_data, selected_tomo_name
    ):
        """Handle lattice selection when user clicks on plot points."""
        if not click_data or not lattice_data or not selected_tomo_name:
            return selected_lattices or {}

        if selected_tomo_name not in lattice_data:
            return selected_lattices or {}

        if not selected_lattices:
            selected_lattices = {}
        if selected_tomo_name not in selected_lattices:
            selected_lattices[selected_tomo_name] = []

        try:
            lattice_id = int(click_data["points"][0]["text"])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error getting lattice ID from click data: {e}")
            return selected_lattices or {}

        if lattice_id in selected_lattices[selected_tomo_name]:
            selected_lattices[selected_tomo_name].remove(lattice_id)
        else:
            selected_lattices[selected_tomo_name].append(lattice_id)

        return selected_lattices

    @app.callback(
        Output("store-clicked-point", "data"),
        Output("div-graph-data", "children"),
        Input("graph-picking", "clickData"),
        State("store-clicked-point", "data"),
        State("store-tomogram-data", "data"),
        State("dropdown-tomo", "value"),
        State("store-lattice-data", "data"),
        prevent_initial_call=True,
    )
    def handle_point_selection(
        click_data,
        clicked_points_data,
        tomogram_raw_data,
        selected_tomo_name,
        lattice_data,
    ):
        """Handle point selection logic for geometric measurements."""
        if not click_data or not selected_tomo_name:
            return clicked_points_data or [], ""

        if lattice_data and selected_tomo_name in lattice_data:
            return clicked_points_data or [], ""

        new_point = extract_point_data(click_data, POSITION_KEYS, ORIENTATION_KEYS)

        if not clicked_points_data:
            clicked_points_data = []
        if len(clicked_points_data) >= 2:
            clicked_points_data = []
        clicked_points_data.append(new_point)

        # If we now have exactly 2 points, calculate parameters
        if len(clicked_points_data) == 2:
            params_message, error = calculate_geometric_params(
                clicked_points_data, particle_from_point_data, html
            )
            if error:
                # Remove the second point if they're too close
                clicked_points_data = clicked_points_data[:-1]
                return clicked_points_data, ""
            return clicked_points_data, params_message

        return clicked_points_data, ""

    @app.callback(
        Output("store-camera", "data"),
        Input("graph-picking", "relayoutData"),
        State("store-camera", "data"),
    )
    def save_camera_position(relayout_data, previous_camera):
        if relayout_data and CAMERA_KEY in relayout_data:
            return relayout_data[CAMERA_KEY]
        else:
            return previous_camera

    @app.callback(
        Output("store-cache-cleared", "data"),
        Input("store-lattice-data", "data"),
        State("store-session-key", "data"),
        State("store-tomogram-data", "data"),
        State("store-selected-lattices", "data"),
        State("dropdown-tomo", "value"),
        State("switch-cone-plot", "on"),
        State("inp-cone-size", "value"),
        State("switch-show-removed", "on"),
        State("switch-flip-particles", "on"),
        State("store-flips", "data"),
        prevent_initial_call=True,
    )
    def handle_lattice_data_cache_update(
        lattice_data,
        session_key,
        tomogram_raw_data,
        selected_lattices,
        selected_tomo_name,
        make_cones,
        cone_size,
        show_removed,
        flip_particles,
        flip_data,
    ):
        """Update cache when lattice data is produced via cleaning"""
        if not session_key or not tomogram_raw_data or not selected_tomo_name:
            return True

        if not make_cones:
            cone_size = -1

        # Clear cache and regenerate with new lattice data
        clear_cache(session_key)

        try:
            data_path = tomogram_raw_data["__data_path__"]
            preload_tomograms(
                selected_tomo_name,
                tomogram_raw_data["__tomogram_names__"],
                data_path,
                session_key,
                lattice_data,
                cone_size,
                show_removed,
                flip_data,
            )
        except Exception as e:
            logger.warning(f"Failed to update cache with new lattice data: {e}")

        return True
