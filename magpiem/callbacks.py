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

from .cleaner import Cleaner, save_cleaning_parameters
from .processing_utils import process_single_tomogram
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
    process_uploaded_file,
    load_previous_session,
    validate_save_inputs,
    extract_particle_ids_for_saving,
    save_file_by_type,
)
from .plotting_helpers import (
    add_selected_points_trace,
    update_lattice_trace_colors,
    get_tomogram_figure,
    apply_figure_customizations,
    extract_point_data,
    calculate_geometric_params,
)
from .plot_cache import (
    _get_or_create_cache_entry,
    _add_to_cache_and_evict,
    clear_cache,
)
from .utilities import validate_required_data, log_callback_start, handle_callback_error

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

        fig = get_tomogram_figure(
            selected_tomo_name,
            tomogram_raw_data,
            session_key,
            lattice_data,
            make_cones,
            cone_size,
            show_removed,
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
        tomogram_raw_data, error = process_uploaded_file(
            filename,
            contents,
            num_images,
            temp_file_dir,
            save_dash_upload,
            get_tomogram_names,
        )
        if error:
            return error, True, {}, {}, {}

        lattice_data, selected_lattices = {}, {}
        session_result, session_error = load_previous_session(
            previous_filename,
            previous_contents,
            tomogram_raw_data,
            temp_file_dir,
            ctx,
            save_dash_upload,
            read_previous_progress,
        )

        if session_error:
            return session_error, True, {}, {}, {}
        elif session_result:
            lattice_data, selected_lattices = session_result

        return (
            f"Found {len(tomogram_raw_data['__tomogram_names__'])} tomograms (loading on-demand)",
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

        tomo_names = tomogram_raw_data["__tomogram_names__"]
        data_path = tomogram_raw_data["__data_path__"]
        total_tomos = len(tomo_names)

        logger.info("Loading .mat file...")
        full_geom = read_emc_mat(data_path)
        if full_geom is None:
            logger.error("Failed to load .mat file")
            return True, True, False, {}, 0

        logger.info(f"Processing {total_tomos} tomograms...")

        lattice_data = {}
        clean_count = 0
        clean_total_time = 0

        for tomo_name in tomo_names:
            clean_count += 1
            lattice_result, processing_time = process_single_tomogram(
                tomo_name,
                full_geom,
                clean_params,
                set_progress,
                clean_count,
                total_tomos,
                clean_total_time,
                read_emc_tomogram_raw_data,
                clean_tomo_with_cpp,
            )

            if lattice_result is not None:
                lattice_data[tomo_name] = lattice_result
                clean_total_time += processing_time

        logger.info("Saving cleaning parameters")
        save_cleaning_parameters(
            dist_goal,
            dist_tol,
            ori_goal,
            ori_tol,
            curv_goal,
            curv_tol,
            cc_thresh,
            min_neighbours,
            array_size,
            allow_flips,
            filename,
            cleaning_params_dir,
        )

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
        # Validate inputs
        is_valid, error_msg = validate_save_inputs(
            output_name, input_name, lattice_data
        )
        if not is_valid:
            logger.warning(error_msg)
            return None, False

        # Extract particle IDs for saving
        saving_ids = extract_particle_ids_for_saving(
            lattice_data, selected_lattices, keep_selected
        )

        # Save file and validate if necessary
        save_success, validation_error = save_file_by_type(
            saving_ids,
            output_name,
            input_name,
            temp_file_dir,
            write_emc_mat,
            write_relion_star,
            validate_mat_files,
        )
        if not save_success:
            return None, True

        out_file = temp_file_dir + output_name
        logger.info("Saving output file: %s", out_file)
        return dcc.send_file(out_file), False
