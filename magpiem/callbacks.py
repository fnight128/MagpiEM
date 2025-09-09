# -*- coding: utf-8 -*-
"""
Callback functions for the MagpiEM Dash application.
"""

import base64
import datetime
import logging
import os
import uuid
from pathlib import Path
from time import time

import diskcache
import numpy as np
import plotly.graph_objects as go
import yaml
from dash import State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output
from dash.long_callback import DiskcacheLongCallbackManager

from .cleaner import Cleaner
from .particle import Particle
from .tomogram import Tomogram
from .cpp_integration import clean_tomo_with_cpp, check_cpp_availability
from .layout import EMPTY_FIG
from .test_mat_file_comparison import validate_mat_files
from .read_write import (
    get_tomogram_names,
    load_single_tomogram_raw_data,
    read_emc_mat,
    read_emc_tomogram_raw_data,
    write_emc_mat,
    write_relion_star,
)
from .plotting_helpers import (
    add_selected_points_trace,
    update_lattice_trace_colors,
)
from .plot_cache import (
    _get_or_create_cache_entry,
    _add_to_cache_and_evict,
    clear_cache,
)

# Constants
CAMERA_KEY = "scene.camera"
POSITION_KEYS = ["x", "y", "z"]
ORIENTATION_KEYS = ["u", "v", "w"]
TEMP_TRACE_NAME = "selected_particle_trace"

logger = logging.getLogger(__name__)


def register_callbacks(app, cache_functions, temp_file_dir, cleaning_params_dir):
    """Register all callbacks with the Dash app."""

    # Unpack cache functions
    get_cached_tomogram_figure = cache_functions["get_cached_tomogram_figure"]
    preload_tomograms = cache_functions["preload_tomograms"]

    @app.callback(
        Output("confirm-cpp-unavailable", "displayed"),
        Input("div-page-load", "children"),
        prevent_initial_call=False,
    )
    def check_cpp_on_startup(_):
        """Check C++ availability on app startup and show warning if unavailable."""
        if check_cpp_availability():
            logger.info("C++ library available")
            return False
        else:
            logger.error("C++ library not available - using Python fallback")
            return True

    @app.callback(
        Output("store-session-key", "data"),
        Input("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def generate_session_key(filename):
        """Generate a unique session key for this user session."""
        return str(uuid.uuid4())

    @app.callback(
        Output("upload-data", "children"),
        Output("input-save-filename", "value"),
        Input("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def update_output(filename):
        if not filename:
            return "Choose File", ""
        else:
            return filename, "out_" + filename

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
        State("store-session-key", "data"),
        State("store-tomogram-data", "data"),
        State("store-lattice-data", "data"),
        State("store-selected-lattices", "data"),
        State("dropdown-tomo", "value"),
        prevent_initial_call=True,
    )
    def handle_plot_style_changes(
        make_cones,
        cone_size,
        cone_clicks,
        show_removed,
        session_key,
        tomogram_raw_data,
        lattice_data,
        selected_lattices,
        selected_tomo_name,
    ):
        """Handle plot style changes and update cache accordingly."""
        logger.debug("handle_plot_style_changes called")
        logger.debug(
            "make_cones=%s, cone_size=%s, show_removed=%s",
            make_cones,
            cone_size,
            show_removed,
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
    ):
        """Retrieve and plot cached tomo"""
        logger.debug("plot_tomo called with selected_tomo_name=%s", selected_tomo_name)
        logger.debug("make_cones=%s, cone_size=%s", make_cones, cone_size)
        logger.debug("cache_cleared=%s", cache_cleared)

        if not selected_tomo_name or not tomogram_raw_data:
            logger.debug("Returning EMPTY_FIG - missing data")
            return EMPTY_FIG

        if selected_tomo_name not in tomogram_raw_data["__tomogram_names__"]:
            logger.debug("Returning EMPTY_FIG - tomogram not found")
            return EMPTY_FIG

        # Get the cached figure
        data_path = tomogram_raw_data["__data_path__"]
        logger.debug("Getting cached figure for %s", selected_tomo_name)

        # Determine cone size based on switch state
        actual_cone_size = cone_size if make_cones else -1
        logger.debug("actual_cone_size=%s", actual_cone_size)

        fig = get_cached_tomogram_figure(
            selected_tomo_name,
            data_path,
            session_key,
            lattice_data,
            actual_cone_size,
            show_removed,  # Use current show_removed state
        )

        logger.debug("Got figure from cache: %s", fig is not None)

        if fig is None:
            logger.debug("Figure is None, returning EMPTY_FIG")
            fig = EMPTY_FIG
        else:
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
                fig = add_selected_points_trace(
                    fig, clicked_point_data, TEMP_TRACE_NAME
                )

        # Restore camera position
        if camera_data:
            fig["layout"]["scene"]["camera"] = camera_data

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

        current_point_data = click_data["points"][0]
        particle_data_keys = POSITION_KEYS + ORIENTATION_KEYS
        new_point = {key: current_point_data[key] for key in particle_data_keys}

        if not clicked_points_data:
            clicked_points_data = []
        if len(clicked_points_data) >= 2:
            clicked_points_data = []
        clicked_points_data.append(new_point)

        # If we now have exactly 2 points, calculate parameters
        if len(clicked_points_data) == 2:
            selected_particles = []
            for idx, point_data in enumerate(clicked_points_data):
                selected_particles.append(particle_from_point_data(point_data, idx=idx))

            # Check if points are too close together - implies same point and could lead to errors
            if selected_particles[0].distance_sq(selected_particles[1]) < 0.001:
                # Remove the second point if they're too close
                clicked_points_data = clicked_points_data[:-1]
                return clicked_points_data, ""

            params_dict = selected_particles[0].calculate_params(selected_particles[1])
            params_message = []
            for param_name, param_value in params_dict.items():
                params_message.append(f"{param_name}: {param_value:.2f}")
                params_message.append(html.Br())

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
        Output("dropdown-filetype", "value"),
        Input("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def update_filetype_dropdown(filename):
        if not filename:
            raise PreventUpdate
        try:
            ext = Path(filename).suffix
        except Exception:
            raise PreventUpdate
        return ext

    @app.callback(
        Output("label-keep-particles", "children"),
        Input("switch-keep-particles", "on"),
        prevent_initial_call=True,
    )
    def update_keeping_particles(keeping):
        if keeping:
            return "Keep selected particles"
        else:
            return "Keep unselected particles"

    @app.callback(
        Output("button-set-cone-size", "n_clicks"),
        Input("div-null", "style"),
        prevent_initial_call=True,
    )
    def cone_clicks(_):
        return 1

    @app.callback(
        Output("download-file", "data"),
        Output("confirm-cant-save-progress", "displayed"),
        Input("button-save-progress", "n_clicks"),
        State("upload-data", "filename"),
        State("slider-num-images", "value"),
        State("store-tomogram-data", "data"),
        State("store-lattice-data", "data"),
        State("store-selected-lattices", "data"),
        prevent_initial_call=True,
    )
    def save_current_progress(
        clicks, filename, num_images, tomogram_raw_data, lattice_data, selected_lattices
    ):
        if not clicks:
            return None, False

        if num_images != 2:
            return None, True

        if not tomogram_raw_data or not lattice_data:
            return None, True

        file_path = temp_file_dir + filename + "_progress.yml"

        tomo_dict = {}
        for tomo_name in tomogram_raw_data["__tomogram_names__"]:
            if tomo_name in lattice_data:
                tomo_progress = {
                    "lattice_data": lattice_data[tomo_name],
                    "selected_lattices": (
                        selected_lattices.get(tomo_name, [])
                        if selected_lattices
                        else []
                    ),
                    "raw_data_loaded": True,
                    "cleaning_completed": True,
                }
                tomo_dict[tomo_name] = tomo_progress

        logger.debug("Saving keys: %s", list(tomo_dict.keys()))
        logger.debug("Saving progress data")
        prog = yaml.safe_dump(tomo_dict)
        with open(file_path, "w") as yaml_file:
            yaml_file.write(prog)
        return dcc.send_file(file_path), False

    @app.callback(
        Output("button-read", "disabled"),
        Input("upload-previous-session", "filename"),
        prevent_initial_call=True,
    )
    def hide_read_button(_):
        return True

    @app.callback(
        Output("collapse-upload", "is_open"),
        Output("collapse-clean", "is_open"),
        Output("collapse-graph-control", "is_open"),
        Output("collapse-save", "is_open"),
        Input("dropdown-tomo", "disabled"),
        State("store-tomogram-data", "data"),
        State("store-lattice-data", "data"),
        prevent_initial_callback=True,
    )
    def open_cards(_, tomogram_raw_data, lattice_data):
        upload_phase = True, False, False, False
        cleaning_phase = False, True, True, False
        saving_phase = False, False, True, True
        if not tomogram_raw_data:
            return upload_phase
        elif not lattice_data:
            return cleaning_phase
        else:
            return cleaning_phase

    @app.callback(
        Output("dropdown-tomo", "options"),
        Output("dropdown-tomo", "value"),
        Output("store-clicked-point", "data"),
        State("dropdown-tomo", "value"),
        Input("dropdown-tomo", "disabled"),
        Input("button-next-Tomogram", "n_clicks"),
        Input("button-previous-Tomogram", "n_clicks"),
        State("store-lattice-data", "data"),
        State("store-tomogram-data", "data"),
        State("store-session-key", "data"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def update_dropdown(
        current_val,
        disabled,
        _,
        __,
        lattice_data,
        tomogram_raw_data,
        session_key,
        filename,
    ):
        tomo_keys = tomogram_raw_data["__tomogram_names__"]

        logger.debug("Available tomograms: %s", tomo_keys)
        if not tomo_keys:
            return [], ""

        if not current_val:
            current_val = tomo_keys[0]

        increment = 0
        if ctx.triggered_id == "button-next-Tomogram":
            increment = 1
        elif ctx.triggered_id == "button-previous-Tomogram":
            increment = -1
        chosen_index = tomo_keys.index(current_val) + increment

        if chosen_index < 0:
            chosen_index = len(tomo_keys) - 1
        elif chosen_index >= len(tomo_keys):
            chosen_index = 0

        chosen_tomo = tomo_keys[chosen_index]

        return tomo_keys, chosen_tomo, {}

    def read_previous_progress(progress_file, tomogram_raw_data):
        """
        Read previous progress from a YAML file and return the data for store components.

        Parameters
        ----------
        progress_file : str
            Path to the progress YAML file
        tomogram_raw_data : dict
            Current tomogram raw data to validate against

        Returns
        -------
        tuple
            (lattice_data, selected_lattices) or error message
        """
        try:
            with open(progress_file, "r") as prev_yaml:
                prev_yaml = yaml.safe_load(prev_yaml)
        except yaml.YAMLError:
            return "Previous session file unreadable"

        geom_keys = set(tomogram_raw_data["__tomogram_names__"])
        prev_keys = set(prev_yaml.keys())
        prev_keys.discard(".__cleaning_parameters__.")

        if not geom_keys == prev_keys:
            if len(prev_keys) in {1, 5}:
                return [
                    "Keys do not match up between previous session and .mat file.",
                    html.Br(),
                    "Previous session only contains {0} keys, did you save the session with only {0} Tomogram(s) "
                    "loaded?".format(len(prev_keys)),
                ]
            geom_missing = list(prev_keys.difference(geom_keys))
            geom_msg = ""
            if len(geom_missing) > 0:
                geom_msg = "Keys present in previous session but not .mat: {}".format(
                    ",".join(geom_missing)
                )
            prev_missing = list(geom_keys.difference(prev_keys))
            prev_msg = ""
            if len(prev_missing) > 0:
                prev_msg = "Keys present in previous session but not .mat: {}".format(
                    ",".join(prev_missing)
                )
            return [
                "Keys do not match up between previous session and .mat file.",
                html.Br(),
                geom_msg,
                html.Br(),
                prev_msg,
            ]

        lattice_data = {}
        selected_lattices = {}

        for tomo_name in geom_keys:
            if tomo_name in prev_yaml:
                tomo_progress = prev_yaml[tomo_name]

                if "lattice_data" in tomo_progress:
                    lattice_data[tomo_name] = tomo_progress["lattice_data"]

                if "selected_lattices" in tomo_progress:
                    selected_lattices[tomo_name] = tomo_progress["selected_lattices"]

        return lattice_data, selected_lattices

    @app.callback(
        Output("label-read", "children"),
        Output("dropdown-tomo", "disabled"),
        Output("store-lattice-data", "data"),
        Output("store-tomogram-data", "data"),
        Output("store-selected-lattices", "data"),
        Input("button-read", "n_clicks"),
        Input("upload-previous-session", "filename"),
        Input("upload-previous-session", "contents"),
        State("upload-data", "filename"),
        State("upload-data", "contents"),
        State("slider-num-images", "value"),
        long_callback=True,
        prevent_initial_call=True,
    )
    def read_tomograms(
        _, previous_filename, previous_contents, filename, contents, num_images
    ):
        if not filename:
            return "Please choose a particle database", True, {}, {}, {}

        num_img_dict = {0: 1, 1: 5, 2: -1}
        num_images = num_img_dict[num_images]

        save_dash_upload(filename, contents, temp_file_dir)

        data_file_path = temp_file_dir + filename

        all_tomogram_names = get_tomogram_names(data_file_path, num_images=num_images)

        if not all_tomogram_names:
            return "Data file Unreadable", True, {}, {}, {}

        tomogram_raw_data = {
            "__tomogram_names__": all_tomogram_names,
            "__data_path__": data_file_path,
            "__num_images__": num_images,
        }
        lattice_data = {}
        selected_lattices = {}

        if ctx.triggered_id == "upload-previous-session":
            save_dash_upload(previous_filename, previous_contents, temp_file_dir)
            progress_path = temp_file_dir + previous_filename
            progress_result = read_previous_progress(progress_path, tomogram_raw_data)

            if isinstance(progress_result, str) or isinstance(progress_result, list):
                return progress_result, True, {}, {}, {}
            else:
                lattice_data, selected_lattices = progress_result

        return (
            f"Found {len(all_tomogram_names)} tomograms (loading on-demand)",
            False,
            lattice_data,
            tomogram_raw_data,
            selected_lattices,
        )

    @app.callback(
        Output("inp-dist-goal", "value"),
        Output("inp-dist-tol", "value"),
        Output("inp-ori-goal", "value"),
        Output("inp-ori-tol", "value"),
        Output("inp-curv-goal", "value"),
        Output("inp-curv-tol", "value"),
        Output("inp-min-neighbours", "value"),
        Output("inp-cc-thresh", "value"),
        Output("inp-array-size", "value"),
        Output("switch-allow-flips", "on"),
        Input("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def read_previous_clean_params(filename):
        if not filename:
            raise PreventUpdate

        base_name = Path(filename).stem
        clean_yaml_name = f"{base_name}_clean_params.yml"

        clean_keys = [
            "distance",
            "distance tolerance",
            "orientation",
            "orientation tolerance",
            "curvature",
            "curvature tolerance",
            "cc threshold",
            "min neighbours",
            "min array size",
            "allow flips",
        ]
        try:
            with open(cleaning_params_dir + clean_yaml_name, "r") as prev_yaml:
                prev_yaml = yaml.safe_load(prev_yaml)
                prev_vals = [prev_yaml[key] for key in clean_keys]
                return prev_vals
        except (FileNotFoundError, yaml.YAMLError, KeyError):
            logger.info(
                f"Couldn't find or read a previous cleaning file for {filename} at {cleaning_params_dir}."
            )
            raise PreventUpdate

    def save_dash_upload(filename, contents, temp_file_dir):
        logger.info("Uploading file: %s", filename)
        data = contents.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(temp_file_dir, filename), "wb") as fp:
            fp.write(base64.decodebytes(data))

    @app.callback(
        Output("progress-processing", "value"),
        Input("interval-processing", "n_intervals"),
        State("store-cleaning-progress", "data"),
    )
    def update_progress(_, progress_data):
        """Update progress bar from stored progress data."""
        return progress_data if progress_data else 0

    @app.callback(
        Output("button-next-Tomogram", "disabled"),
        Output("collapse-clean", "is_open"),
        Output("collapse-save", "is_open"),
        Output("store-lattice-data", "data"),
        Output("store-cleaning-progress", "data"),
        State("inp-dist-goal", "value"),
        State("inp-dist-tol", "value"),
        State("inp-ori-goal", "value"),
        State("inp-ori-tol", "value"),
        State("inp-curv-goal", "value"),
        State("inp-curv-tol", "value"),
        State("inp-min-neighbours", "value"),
        State("inp-cc-thresh", "value"),
        State("inp-array-size", "value"),
        State("switch-allow-flips", "on"),
        State("store-tomogram-data", "data"),
        State("store-session-key", "data"),
        State("upload-data", "filename"),
        Input("button-full-clean", "n_clicks"),
        prevent_initial_call=True,
        background=True,
        # Disable buttons during cleaning
        running=[
            (Output("button-full-clean", "disabled"), True, False),
            (
                Output("progress-processing", "style"),
                {"visibility": "visible"},
                {"visibility": "hidden"},
            ),
        ],
        progress=Output("store-cleaning-progress", "data"),
    )
    def run_cleaning(
        set_progress: callable,
        dist_goal: float,
        dist_tol: float,
        ori_goal: float,
        ori_tol: float,
        curv_goal: float,
        curv_tol: float,
        min_neighbours: int,
        cc_thresh: float,
        array_size: int,
        allow_flips: bool,
        tomogram_raw_data: dict,
        session_key: str,
        filename: str,
        clicks,
    ):
        if not clicks:
            return True, True, False, {}, 0

        if not tomogram_raw_data:
            logger.warning("No tomogram data available for cleaning")
            return True, True, False, {}, 0

        # Clear cache when cleaning starts since all cached figures will need to be
        # replotted with lattice data
        if session_key:
            clear_cache(session_key)

        clean_params = Cleaner.from_user_params(
            cc_thresh,
            min_neighbours,
            array_size,
            dist_goal,
            dist_tol,
            ori_goal,
            ori_tol,
            curv_goal,
            curv_tol,
            allow_flips,
        )

        logger.info("Starting cleaning process")

        lattice_data = {}

        clean_count = 0
        clean_total_time = 0

        tomo_names = tomogram_raw_data["__tomogram_names__"]
        data_path = tomogram_raw_data["__data_path__"]
        total_tomos = len(tomo_names)

        logger.info("Loading .mat file...")
        full_geom = read_emc_mat(data_path)
        if full_geom is None:
            logger.error("Failed to load .mat file")
            return True, True, False, {}, 0

        logger.info(f"Processing {total_tomos} tomograms...")

        for tomo_name in tomo_names:
            t0 = time()
            logger.debug(f"Cleaning tomogram: {tomo_name}")

            if tomo_name not in full_geom:
                logger.warning(f"Tomogram {tomo_name} not found in .mat file")
                continue

            tomo_raw_data = read_emc_tomogram_raw_data(full_geom[tomo_name], tomo_name)
            if tomo_raw_data is None:
                logger.warning(f"Failed to extract data for tomogram: {tomo_name}")
                continue

            lattice_data[tomo_name] = clean_tomo_with_cpp(tomo_raw_data, clean_params)
            clean_total_time += time() - t0
            clean_count += 1

            progress_percent = int((clean_count / total_tomos) * 100)
            set_progress(progress_percent)

            # Only calculate time remaining every 10 tomograms to reduce overhead
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

        logger.info("Saving cleaning parameters")
        cleaning_params_dict = {
            "distance": dist_goal,
            "distance tolerance": dist_tol,
            "orientation": ori_goal,
            "orientation tolerance": ori_tol,
            "curvature": curv_goal,
            "curvature tolerance": curv_tol,
            "cc threshold": cc_thresh,
            "min neighbours": min_neighbours,
            "min array size": array_size,
            "allow flips": allow_flips,
        }

        if filename:
            base_name = Path(filename).stem
            clean_yaml_name = f"{base_name}_clean_params.yml"
        else:
            clean_yaml_name = "unknown_file_clean_params.yml"

        with open(cleaning_params_dir + clean_yaml_name, "w") as yaml_file:
            yaml_file.write(yaml.safe_dump(cleaning_params_dict))

        set_progress(100)
        return False, False, True, lattice_data, 100

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
            )
        except Exception as e:
            logger.warning(f"Failed to update cache with new lattice data: {e}")

        return True

    @app.callback(
        Output("download-file", "data"),
        Output("confirm-validation-failed", "displayed"),
        State("input-save-filename", "value"),
        State("upload-data", "filename"),
        State("switch-keep-particles", "on"),
        State("checklist-save-additional", "value"),
        State("store-selected-lattices", "data"),
        State("store-lattice-data", "data"),
        Input("button-save", "n_clicks"),
        prevent_initial_call=True,
        long_callback=True,
    )
    def save_result(
        output_name,
        input_name,
        keep_selected,
        save_additional,
        selected_lattices,
        lattice_data,
        _,
    ):
        if not output_name:
            return None
        if output_name == input_name:
            logger.warning("Output and input file cannot be identical")
            return None

        if not lattice_data:
            logger.warning("No lattice data available for saving")
            return None
        saving_ids = {}
        for tomo_name, tomo_lattice_data in lattice_data.items():
            if not tomo_lattice_data:
                continue

            tomo_selected_lattices = (
                selected_lattices.get(tomo_name, []) if selected_lattices else []
            )

            particle_ids = []
            for lattice_id, particle_indices in tomo_lattice_data.items():
                lattice_id_int = (
                    int(lattice_id) if isinstance(lattice_id, str) else lattice_id
                )

                lattice_is_selected = lattice_id_int in tomo_selected_lattices
                should_include = (
                    lattice_is_selected if keep_selected else not lattice_is_selected
                )

                if should_include:
                    particle_ids.extend(particle_indices)

            # Only add tomogram to saving_ids if it has particles to save
            if particle_ids:
                saving_ids[tomo_name] = particle_ids

        if ".mat" in input_name:
            write_emc_mat(
                saving_ids,
                temp_file_dir + output_name,
                temp_file_dir + input_name,
            )

            # Ensure .mat files are otherwise identical
            out_file = temp_file_dir + output_name
            input_file = temp_file_dir + input_name
            logger.info("Running validation test on output file: %s", out_file)

            try:
                validate_mat_files(input_file, out_file)
            except Exception as e:
                logger.error("File validation failed: %s", str(e))
                # Trigger error popup
                return None, True

        elif ".star" in input_name:
            write_relion_star(
                saving_ids,
                temp_file_dir + output_name,
                temp_file_dir + input_name,
            )

        out_file = temp_file_dir + output_name
        logger.info("Saving output file: %s", out_file)
        return dcc.send_file(out_file), False
