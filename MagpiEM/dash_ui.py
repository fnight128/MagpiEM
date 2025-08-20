# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:53:45 2022

@author: Frank
"""
import colorsys
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import base64
import math
import yaml
from pathlib import Path
import ctypes

import webbrowser

import glob
import datetime
from time import time

from dash import dcc, html, State, ctx
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
import dash_daq as daq

from flask import Flask

from .classes import Particle, Tomogram, Cleaner, simple_figure
from .read_write import (
    read_relion_star,
    read_multiple_tomograms,
    read_multiple_tomograms_raw_data,
    write_relion_star,
    write_emc_mat,
    read_tomo_names,
    read_single_tomogram,
)
from .plotting_helpers import (
    create_particle_plot_from_raw_data,
    create_lattice_plot_from_raw_data,
    create_scatter_trace,
)

WHITE = "#FFFFFF"
GREY = "#646464"
BLACK = "#000000"

__dash_tomograms = dict()
EMPTY_FIG = simple_figure()

__progress = 0.0

TEMP_FILE_DIR = "static/"
CLEAN_YAML_NAME = "prev_clean_params.yml"

CAMERA_KEY = "scene.camera"

POSITION_KEYS = ["x", "y", "z"]
ORIENTATION_KEYS = ["u", "v", "w"]

TEMP_TRACE_NAME = "selected_particle_trace"


# Define the CleanParams struct for ctypes
class CleanParams(ctypes.Structure):
    _fields_ = [
        ("min_distance", ctypes.c_float),
        ("max_distance", ctypes.c_float),
        ("min_orientation", ctypes.c_float),
        ("max_orientation", ctypes.c_float),
        ("min_curvature", ctypes.c_float),
        ("max_curvature", ctypes.c_float),
        ("min_lattice_size", ctypes.c_uint),
        ("min_neighbours", ctypes.c_uint),
    ]


def setup_cpp_library() -> ctypes.CDLL:
    """Load and configure the C++ library"""
    # Get the path to the processing directory relative to this file
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    libname = project_root / "processing" / "processing.dll"

    print("Loading processing library from:", libname)
    print("Current file:", current_file)
    print("Project root:", project_root)

    if not libname.exists():
        # Try alternative paths
        alt_paths = [
            project_root / "processing.dll",
            Path.cwd() / "processing" / "processing.dll",
            Path.cwd() / "processing.dll",
        ]

        for alt_path in alt_paths:
            if alt_path.exists():
                libname = alt_path
                print(f"Found DLL at alternative path: {libname}")
                break
        else:
            raise FileNotFoundError(
                f"Processing DLL not found. Tried:\n- {libname}\n"
                + "\n".join(f"- {p}" for p in alt_paths)
            )

    c_lib = ctypes.CDLL(str(libname))

    # Set up function signatures
    c_lib.clean_particles.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(CleanParams),
        ctypes.POINTER(ctypes.c_int),
    ]
    c_lib.clean_particles.restype = None

    return c_lib


def convert_raw_data_to_cpp_format(tomogram_raw_data: list) -> tuple[np.ndarray, int]:
    """
    Convert raw tomogram data to the format expected by the C++ library.

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]

    Returns
    -------
    tuple[np.ndarray, int]
        Flattened data array and number of particles
    """
    if not tomogram_raw_data:
        return np.array([]), 0

    # Flatten the data: [[x,y,z], [rx,ry,rz]] -> [x,y,z,rx,ry,rz]
    flat_data = []
    for particle in tomogram_raw_data:
        pos, orient = particle
        # Convert both to lists to ensure consistent behavior
        pos_list = list(pos)
        orient_list = list(orient)
        flat_data.extend(pos_list + orient_list)

    return np.array(flat_data, dtype=np.float32), len(tomogram_raw_data)


def clean_tomo_with_cpp(tomogram_raw_data: list, clean_params: Cleaner) -> dict:
    """
    Clean tomogram data using the C++ library.

    Parameters
    ----------
    tomogram_raw_data : list
        Raw particle data in format [[[x,y,z], [rx,ry,rz]], ...]
    clean_params : Cleaner
        Cleaning parameters

    Returns
    -------
    dict
        Dictionary with particle_id -> lattice_id mappings
    """
    # Convert data to C++ format
    flat_data, num_particles = convert_raw_data_to_cpp_format(tomogram_raw_data)

    if num_particles == 0:
        return {}

    # Create C++ parameters structure
    cpp_params = CleanParams(
        min_distance=clean_params.dist_range[0],
        max_distance=clean_params.dist_range[1],
        min_orientation=clean_params.ori_range[0],
        max_orientation=clean_params.ori_range[1],
        min_curvature=clean_params.curv_range[0],
        max_curvature=clean_params.curv_range[1],
        min_lattice_size=clean_params.min_lattice_size,
        min_neighbours=clean_params.min_neighbours,
    )

    # Create C arrays
    c_array = (ctypes.c_float * len(flat_data))(*flat_data)
    results_array = (ctypes.c_int * num_particles)()

    # Load and call C++ library
    c_lib = setup_cpp_library()
    c_lib.clean_particles(
        c_array, num_particles, ctypes.byref(cpp_params), results_array
    )

    # Convert results back to dictionary format
    lattice_assignments = {}
    for i in range(num_particles):
        lattice_id = int(results_array[i])
        if lattice_id not in lattice_assignments:
            lattice_assignments[lattice_id] = []
        lattice_assignments[lattice_id].append(i)

    return lattice_assignments


def main(open_browser=True):
    """
    Main function to run the MagpiEM Dash application.

    Parameters
    ----------
    open_browser : bool, optional
        Whether to automatically open the application in the default browser.
        Default is True. Set to False for testing or headless operation.
    """
    server = Flask(__name__)
    app = DashProxy(
        server=server,
        external_stylesheets=[dbc.themes.SOLAR],
        transforms=[MultiplexerTransform()],
        title="MagpiEM",
        update_title=None,  # Prevent "Updating..." title
    )
    load_figure_template("SOLAR")

    if not os.path.exists(TEMP_FILE_DIR):
        os.makedirs(TEMP_FILE_DIR)

    __dash_tomograms = dict()

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
        # Use inputs to re-plot when changes are made to data, rather than
        # keeping all data changes in this function.
        Output("graph-picking", "figure"),
        Input("dropdown-tomo", "value"),
        Input("store-lattice-data", "data"),
        State("store-tomogram-data", "data"),
        Input(
            "store-selected-lattices", "data"
        ),
        Input("switch-cone-plot", "on"),
        State("inp-cone-size", "value"),
        Input("button-set-cone-size", "n_clicks"),
        Input("switch-show-removed", "on"),
        State("store-clicked-point", "data"),
        State("store-camera", "data"),
        prevent_initial_call=True,
    )
    def plot_tomo(
        selected_tomo_name,
        lattice_data,
        tomogram_raw_data,
        selected_lattices,
        make_cones: bool,
        cone_size: float,
        _,
        show_removed: bool,
        clicked_point_data,
        camera_data,
    ):
        if not make_cones:
            cone_size = -1

        # must always return a graph object or can break dash
        if not selected_tomo_name:
            return EMPTY_FIG

        # Create the main plot
        if tomogram_raw_data and selected_tomo_name in tomogram_raw_data:
            current_tomo_raw_data = tomogram_raw_data[selected_tomo_name]

            # Check if cleaning has been run for this tomogram
            if lattice_data and selected_tomo_name in lattice_data:
                # Get selected lattices for this tomogram
                tomo_selected_lattices = (
                    selected_lattices.get(selected_tomo_name, [])
                    if selected_lattices
                    else []
                )

                # lattice-based plot
                fig = create_lattice_plot_from_raw_data(
                    current_tomo_raw_data,
                    lattice_data[selected_tomo_name],
                    cone_size=cone_size,
                    show_removed_particles=show_removed,
                    selected_lattices=tomo_selected_lattices,
                )
            else:
                # No cleaning data, plot all particles in single colour
                fig = create_particle_plot_from_raw_data(
                    current_tomo_raw_data,
                    cone_size=cone_size,
                    showing_removed_particles=show_removed,
                    opacity=0.8,
                    colour="white",
                )
        else:
            # Fallback to empty figure if no data available
            fig = EMPTY_FIG

        # Add selected points if they exist
        if clicked_point_data:
            selected_points = []

            # Extract point coordinates
            for point_key in ["first_point", "second_point"]:
                if point_key in clicked_point_data:
                    point_data = clicked_point_data[point_key]
                    selected_points.append(
                        [point_data["x"], point_data["y"], point_data["z"]]
                    )

            if selected_points:
                positions = np.array(selected_points)
                particles_scatter_trace = create_scatter_trace(
                    positions, colour="black", opacity=0.8
                )
                particles_scatter_trace.name = TEMP_TRACE_NAME
                particles_scatter_trace.marker.size = (
                    8  # Override default size for selected points
                )
                fig.add_trace(particles_scatter_trace)

        # Restore camera position if available
        if camera_data:
            fig["layout"]["scene"]["camera"] = camera_data

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

        # Initialize selected_lattices if it doesn't exist
        if not selected_lattices:
            selected_lattices = {}
        if selected_tomo_name not in selected_lattices:
            selected_lattices[selected_tomo_name] = []

        # Get the clicked point data
        lattice_id = int(click_data["points"][0]["text"])

        # Toggle the selection for this lattice
        if lattice_id in selected_lattices[selected_tomo_name]:
            selected_lattices[selected_tomo_name].remove(lattice_id)
        else:
            selected_lattices[selected_tomo_name].append(lattice_id)

        return selected_lattices

    @app.callback(
        Output("store-clicked-point", "data"),
        Output("div-graph-data", "children"),
        Output("interval-clear-points", "disabled"),
        Output("store-last-click", "data"),
        Input("graph-picking", "clickData"),
        State("store-clicked-point", "data"),
        State("store-tomogram-data", "data"),
        State("dropdown-tomo", "value"),
        State("store-last-click", "data"),
        prevent_initial_call=True,
    )
    def handle_point_selection(
        click_data, previous_point_data, tomogram_raw_data, selected_tomo_name, last_click_time
    ):
        """Handle point selection logic for geometric measurements."""
        if not click_data or not selected_tomo_name:
            return previous_point_data, "", True, last_click_time

        current_time = time()

        # Cooldown to prevent erroneous clicks
        if current_time - last_click_time < 0.5:
            raise PreventUpdate

        if previous_point_data:
            # Calculate and return relation between particles
            current_point_data = click_data["points"][0]
            selected_particles = []

            # Extract the first point data from the stored structure
            first_point_data = previous_point_data["first_point"]

            for idx, point_data in enumerate([first_point_data, current_point_data]):
                selected_particles.append(particle_from_point_data(point_data, idx=idx))

            if selected_particles[0].distance_sq(selected_particles[1]) < 0.001:
                # Picked the same particle twice
                raise PreventUpdate

            params_dict = selected_particles[0].calculate_params(selected_particles[1])
            params_message = []
            for param_name, param_value in params_dict.items():
                params_message.append(f"{param_name}: {param_value:.2f}")
                params_message.append(html.Br())

            # Store both points temporarily so they can be plotted
            # The plot_tomo callback will show both points, then we'll clear them
            both_points = {
                "first_point": first_point_data,
                "second_point": {
                    "x": current_point_data["x"],
                    "y": current_point_data["y"],
                    "z": current_point_data["z"],
                    "u": current_point_data["u"],
                    "v": current_point_data["v"],
                    "w": current_point_data["w"],
                },
                "clear_after_plot": True,
            }
            return both_points, params_message, False, current_time  # Enable interval to clear points
        else:
            # Store the first point data for later comparison
            point_data = click_data["points"][0]
            particle_data_keys = POSITION_KEYS + ORIENTATION_KEYS
            first_point_data = {key: point_data[key] for key in particle_data_keys}
            return {"first_point": first_point_data}, "", True, current_time

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
        Output("store-clicked-point", "data"),
        Output("interval-clear-points", "disabled"),
        Input("interval-clear-points", "n_intervals"),
        State("store-clicked-point", "data"),
        prevent_initial_call=True,
    )
    def clear_two_point_selection(n_intervals, clicked_point_data):
        """Clear two-point selection after a brief delay to allow plotting."""
        if clicked_point_data and "clear_after_plot" in clicked_point_data:
            # Clear the two-point selection after it's been plotted
            return {}, True  # Disable the interval
        return clicked_point_data, True  # Keep interval disabled

    @app.callback(
        Output("dropdown-filetype", "value"),
        Input("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def update_filetype_dropdown(filename):
        if not filename:
            raise PreventUpdate
        x = Path(filename).suffix
        print(x)
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
        # dummy function to trigger graph updating
        return 1

    @app.callback(
        Output("button-set-cone-size", "n_clicks"),
        State("button-set-cone-size", "n_clicks"),
        State("dropdown-tomo", "value"),
        State("switch-toggle-conv-all", "on"),
        Input("button-toggle-convex", "n_clicks"),
        prevent_initial_call=True,
    )
    def select_convex(clicks, current_tomo, all_tomos, _):
        global __dash_tomograms
        raise NotImplementedError(
            "select_convex function still uses __dash_tomograms and needs to be refactored to use store-tomogram-data"
        )
        if all_tomos:
            for tomo in __dash_tomograms.values():
                tomo.toggle_convex_arrays()
        else:
            __dash_tomograms[current_tomo].toggle_convex_arrays()
        return int(clicks or 0) + 1

    @app.callback(
        Output("button-set-cone-size", "n_clicks"),
        State("button-set-cone-size", "n_clicks"),
        State("dropdown-tomo", "value"),
        State("switch-toggle-conv-all", "on"),
        Input("button-toggle-concave", "n_clicks"),
        prevent_initial_call=True,
    )
    def select_concave(clicks, current_tomo, all_tomos, _):
        # TODO make these a single callback
        global __dash_tomograms
        raise NotImplementedError(
            "select_concave function still uses __dash_tomograms and needs to be refactored to use store-tomogram-data"
        )
        if all_tomos:
            for tomo in __dash_tomograms.values():
                tomo.toggle_concave_arrays()
        else:
            __dash_tomograms[current_tomo].toggle_concave_arrays()
        return int(clicks or 0) + 1

    @app.callback(
        Output("download-file", "data"),
        Output("confirm-cant-save-progress", "displayed"),
        Input("button-save-progress", "n_clicks"),
        State("upload-data", "filename"),
        State("slider-num-images", "value"),
        prevent_initial_call=True,
    )
    def save_current_progress(clicks, filename, num_images):
        if not clicks:
            return None, False

        if num_images != 2:
            return None, True

        file_path = TEMP_FILE_DIR + filename + "_progress.yml"

        global __dash_tomograms
        raise NotImplementedError(
            "save_current_progress function still uses __dash_tomograms and needs to be refactored to use store-tomogram-data"
        )
        tomo_dict = {}
        for name, tomo in __dash_tomograms.items():
            tomo_dict[name] = tomo.write_progress_dict()
        print("Saving keys:", tomo_dict.keys())
        print(tomo_dict)
        prog = yaml.safe_dump(tomo_dict)
        with open(file_path, "w") as yaml_file:
            yaml_file.write(prog)
        return dcc.send_file(file_path), False

    @app.callback(
        Input("upload-previous-session", "filename"),
        Output("button-read", "disabled"),
        prevent_initial_call=True,
    )
    def hide_read_button(_):
        return True

    # desynchronise opening card from processing
    # otherwise looks laggy when opening
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
        State("dropdown-tomo", "value"),
        Input("dropdown-tomo", "disabled"),
        Input("button-next-Tomogram", "n_clicks"),
        Input("button-previous-Tomogram", "n_clicks"),
        State("store-lattice-data", "data"),
        State("store-tomogram-data", "data"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def update_dropdown(
        current_val, disabled, _, __, lattice_data, tomogram_raw_data, filename
    ):

        # unfortunately need to merge two callbacks here, dash does not allow multiple
        # callbacks with the same output so use ctx to distinguish between cases

        tomo_keys = list(tomogram_raw_data.keys())

        print("Available tomograms:", tomo_keys)
        if not tomo_keys:
            return [], ""

        if not current_val:
            current_val = tomo_keys[0]

        # moving to next/prev item in dropdown when next/prev Tomogram button pressed
        increment = 0
        if ctx.triggered_id == "button-next-Tomogram":
            increment = 1
        elif ctx.triggered_id == "button-previous-Tomogram":
            increment = -1
        chosen_index = tomo_keys.index(current_val) + increment

        # allow wrapping around
        if chosen_index < 0:
            chosen_index = len(tomo_keys) - 1
        elif chosen_index >= len(tomo_keys):
            chosen_index = 0

        chosen_tomo = tomo_keys[chosen_index]

        return tomo_keys, chosen_tomo

    def read_uploaded_tomo(data_path, progress, num_images=-1):
        if ".mat" in data_path:
            tomograms = read_multiple_tomograms(data_path, num_images=num_images)
        elif ".star" in data_path:
            raise NotImplementedError("Support for .star files is not yet implemented")
            # tomograms = read_relion_star(data_path, num_images=num_images)
        else:
            return
        return tomograms

    def read_previous_progress(progress_file):
        global __dash_tomograms
        raise NotImplementedError(
            "read_previous_progress function still uses __dash_tomograms and needs to be refactored to use store-tomogram-data"
        )
        try:
            with open(progress_file, "r") as prev_yaml:
                prev_yaml = yaml.safe_load(prev_yaml)
        except yaml.YAMLError:
            return "Previous session file unreadable"

        # check keys line up between files
        geom_keys = set(__dash_tomograms.keys())
        prev_keys = set(prev_yaml.keys())
        prev_keys.discard(".__cleaning_parameters__.")

        if not geom_keys == prev_keys:
            if len(prev_keys) in {1, 5}:
                # likely saved result after only loading few tomograms
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
                    ",".join(geom_missing)
                )
            return [
                "Keys do not match up between previous session and .mat file.",
                html.Br(),
                geom_msg,
                html.Br(),
                prev_msg,
            ]

        raise NotImplementedError(
            "read_previous_progress function still uses __dash_tomograms and needs to be refactored to use store-tomogram-data"
        )
        for tomo_name, tomo in __dash_tomograms.items():
            tomo.apply_progress_dict(prev_yaml[tomo_name])

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
            return "Please choose a particle database", True, {}, {}

        global __progress
        __progress = 0.0

        num_img_dict = {0: 1, 1: 5, 2: -1}
        num_images = num_img_dict[num_images]

        # ensure temp directory clear (but preserve parameter files)
        all_files = glob.glob(TEMP_FILE_DIR + "*")
        # Keep parameter files but remove other temp files
        all_files = [file for file in all_files if not file.endswith("_clean_params.yml")]
        if all_files:
            print("Pre-existing temp files found, removing:", all_files)
            for f in all_files:
                os.remove(f)

        save_dash_upload(filename, contents)

        data_file_path = TEMP_FILE_DIR + filename

        # Read raw data for store
        tomogram_raw_data = read_multiple_tomograms_raw_data(
            data_file_path, num_images=num_images
        )

        if not tomogram_raw_data:
            return "Data file Unreadable", True, {}, {}

        if ctx.triggered_id == "upload-previous-session":
            save_dash_upload(previous_filename, previous_contents)
            progress_path = TEMP_FILE_DIR + previous_filename
            read_previous_progress(progress_path)

        # Initialize lattice_data as empty dictionary - will be populated after cleaning
        lattice_data = {}
        # Initialize selected_lattices as empty dictionary - will be populated when user selects lattices
        selected_lattices = {}

        return (
            "Tomograms read",
            False,
            lattice_data,
            tomogram_raw_data,
            selected_lattices,
        )

    @app.callback(
        Input("upload-data", "filename"),
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
        prevent_initial_call=True,
    )
    def read_previous_clean_params(filename):
        if not filename:
            raise PreventUpdate
            
        global TEMP_FILE_DIR
        # Create filename-specific YAML name
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
            "min " "neighbours",
            "min array size",
            "allow flips",
        ]
        try:
            with open(TEMP_FILE_DIR + clean_yaml_name, "r") as prev_yaml:
                prev_yaml = yaml.safe_load(prev_yaml)
                prev_vals = [prev_yaml[key] for key in clean_keys]
                return prev_vals
        except FileNotFoundError or yaml.YAMLError or KeyError:
            print(f"Couldn't find or read a previous cleaning file for {filename}.")
            raise PreventUpdate

    def clean_tomo(tomo, clean_params):
        tomo.set_clean_params(clean_params)

        tomo.reset_cleaning()

        return tomo.autoclean()

    def save_dash_upload(filename, contents):
        print("Uploading file:", filename)
        data = contents.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(TEMP_FILE_DIR, filename), "wb") as fp:
            fp.write(base64.decodebytes(data))

    @app.callback(
        Output("progress-processing", "value"),
        Input("interval-processing", "n_intervals"),
    )
    def update_progress(_):
        global __progress
        return __progress * 100

    @app.callback(
        Output("button-next-Tomogram", "disabled"),
        Output("collapse-clean", "is_open"),
        Output("collapse-save", "is_open"),
        Output("store-lattice-data", "data"),
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
        State("upload-data", "filename"),
        Input("button-full-clean", "n_clicks"),
        Input("button-preview-clean", "n_clicks"),
        prevent_initial_call=True,
        long_callback=True,
    )
    def run_cleaning(
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
        filename: str,
        clicks,
        clicks2,
    ):
        if not clicks or clicks2:
            return True, True, False, {}

        if not tomogram_raw_data:
            print("No tomogram data available for cleaning")
            return True, True, False, {}

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

        print("Clean")

        global __clean_yaml_name, __progress

        lattice_data = {}

        __progress = 0.0

        clean_count = 0
        clean_total_time = 0
        total_tomos = len(tomogram_raw_data.keys())

        for tomo_name, tomo_raw_data in tomogram_raw_data.items():
            t0 = time()
            print(f"Cleaning tomogram: {tomo_name}")
            lattice_data[tomo_name] = clean_tomo_with_cpp(tomo_raw_data, clean_params)
            print(lattice_data[tomo_name])
            clean_total_time += time() - t0
            clean_count += 1
            tomos_remaining = total_tomos - clean_count
            clean_speed = clean_total_time / clean_count
            secs_remaining = clean_speed * tomos_remaining
            formatted_time_remaining = str(
                datetime.timedelta(seconds=secs_remaining)
            ).split(".")[0]
            __progress = clean_count / total_tomos
            print(f"Time remaining: {formatted_time_remaining}")
            print()

        print("Saving cleaning parameters")
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

        # Create filename-specific YAML name
        if filename:
            base_name = Path(filename).stem
            clean_yaml_name = f"{base_name}_clean_params.yml"
        else:
            clean_yaml_name = "unknown_file_clean_params.yml"

        with open(TEMP_FILE_DIR + clean_yaml_name, "w") as yaml_file:
            yaml_file.write(yaml.safe_dump(cleaning_params_dict))
        return False, False, True, lattice_data

    size = "50px"

    def inp_num(id_suffix):
        return html.Td(
            dbc.Input(
                id="inp-" + id_suffix,
                type="number",
                style={"appearance": "textfield", "width": size},
            ),
            style={"width": size},
        )

    param_table = html.Table(
        [
            html.Tr(
                [
                    html.Th("Parameter"),
                    html.Th("Value", style={"width": "50px"}),
                    html.Th(""),
                    html.Th("Tolerance"),
                ]
            ),
            html.Tr(
                [
                    html.Td("Distance (px)"),
                    inp_num("dist-goal"),
                    html.Td("±"),
                    inp_num("dist-tol"),
                ]
            ),
            html.Tr(
                [
                    html.Td("Orientation (°)"),
                    inp_num("ori-goal"),
                    html.Td("±"),
                    inp_num("ori-tol"),
                ]
            ),
            html.Tr(
                [
                    html.Td("Curvature (°)"),
                    inp_num("curv-goal"),
                    html.Td("±"),
                    inp_num("curv-tol"),
                ]
            ),
            html.Tr([html.Td("Min. Neighbours"), inp_num("min-neighbours")]),
            html.Tr([html.Td("CC Threshold"), inp_num("cc-thresh")]),
            html.Tr([html.Td("Min. Lattice Size"), inp_num("array-size")]),
            html.Tr(
                [
                    html.Td("Allow Flipped Particles"),
                    daq.BooleanSwitch(id="switch-allow-flips", on=False),
                ]
            ),
            html.Tr(
                html.Td(
                    dbc.Button(
                        # disable for now
                        "Preview Cleaning",
                        id="button-preview-clean",
                        color="secondary",
                        style={"display": "none"},
                    ),
                    colSpan=4,
                ),
            ),
            html.Tr(
                html.Td(dbc.Button("Run Cleaning", id="button-full-clean"), colSpan=4),
            ),
        ],
        style={"overflow": "hidden", "margin": "3px", "width": "100%"},
    )

    upload_file = dcc.Upload(
        id="upload-data",
        children="Choose File",
        style={
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "20px",
            "overflow": "hidden",
        },
        multiple=False,
    )

    def card(title: str, contents):
        # width = str(width) + "%"
        return dbc.Card(
            [dbc.CardHeader(title), dbc.CardBody([contents])],
            style={
                "width": "100%",
                "margin": "5px",
                "height": "100%",
                # "padding": "100px"
                # "float": "right",
            },
            # id='card-'
        )

    def collapsing_card(
        display_card: dbc.Card, collapse_id: str, start_open: bool = False
    ):
        return dbc.Collapse(
            display_card,
            id="collapse-" + collapse_id,
            is_open=start_open,
            style={"width": "450px", "height": "100%"},
        )  # ,style={"float":"right"})

    emptydiv = html.Div(
        [
            # nonexistent outputs for callbacks with no visible effect
            html.Div(id="div-null", style={"display": "none"}),
        ],
        style={"display": "none"},
    )

    upload_table = html.Table(
        [
            html.Tr([html.Td(upload_file)]),
            html.Tr(
                dcc.Dropdown(
                    [".mat", ".star"],
                    id="dropdown-filetype",
                    clearable=False,
                )
            ),
            html.Tr([html.Td("Number of Images to Process")]),
            html.Tr(
                [
                    html.Td(
                        dcc.Slider(
                            0,
                            2,
                            step=None,
                            marks={
                                0: "1 Image",
                                1: "5 Images",
                                2: "All Images",
                            },
                            value=2,
                            id="slider-num-images",
                        )
                    )
                ]
            ),
            html.Tr(html.Br()),
            html.Tr(
                [
                    html.Td(
                        dbc.Button(
                            "Read tomograms", id="button-read", style={"width": "100%"}
                        )
                    ),
                ]
            ),
            html.Tr(
                html.Td(
                    dcc.Upload(
                        "Load Previous Session",
                        id="upload-previous-session",
                        multiple=False,
                        style={
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "3px",
                            "textAlign": "center",
                            "margin": "20px",
                            "overflow": "hidden",
                        },
                    )
                )
            ),
            html.Tr(html.Div(id="label-read")),
        ],
        style={
            "overflow": "hidden",
            "margin": "3px",
            "width": "100%",
        },
    )

    graph_controls_table = html.Table(
        [
            html.Tr(
                [
                    html.Td("Tomogram"),
                    dcc.Dropdown(
                        [],
                        id="dropdown-tomo",
                        style={"width": "300px"},
                        clearable=False,
                        disabled=True,
                    ),
                ]
            ),
            html.Tr(
                [
                    html.Td("Cone Plot", id="label-cone-plot"),
                    daq.BooleanSwitch(id="switch-cone-plot", on=True),
                ]
            ),
            html.Tr(
                [
                    html.Td("Overall Cone Size", id="label-cone-size"),
                    html.Td(
                        dbc.Input(
                            id="inp-cone-size",
                            value=10,
                            type="number",
                            style={"width": "70%"},
                        )
                    ),
                    html.Td(dbc.Button("Set", id="button-set-cone-size")),
                ]
            ),
            html.Tr(
                [
                    html.Td(
                        dbc.Button("Toggle Convex", id="button-toggle-convex"),
                    ),
                    html.Td(
                        dbc.Button("Toggle Concave", id="button-toggle-concave"),
                    ),
                    html.Td(
                        [
                            daq.BooleanSwitch(id="switch-toggle-conv-all", on=False),
                            html.Div("All tomos", style={"margin": "auto"}),
                        ],
                        style={"text-align": "center"},
                    ),
                ],
            ),
            html.Tr(
                [
                    html.Td("Show Removed Particles", id="label-show-removed"),
                    daq.BooleanSwitch(id="switch-show-removed", on=False),
                ]
            ),
            html.Tr(
                [
                    html.Td(
                        dbc.Button(
                            "Previous Tomogram",
                            id="button-previous-Tomogram",
                        ),
                    )
                ]
            ),
        ],
        style={
            "overflow": "hidden",
            "margin": "3px",
            "width": "100%",
            "table-layout": "fixed",
        },
    )

    save_table = html.Table(
        [
            html.Tr(
                html.Td(dbc.Button("Save Current Progress", id="button-save-progress"))
            ),
            html.Tr(
                [
                    html.Td("Keep selected particles", id="label-keep-particles"),
                    daq.BooleanSwitch(id="switch-keep-particles", on=True),
                ]
            ),
            html.Tr(
                [
                    html.Td("Save as:"),
                    html.Td(
                        dbc.Input(id="input-save-filename", style={"width": "100px"})
                    ),
                ]
            ),
            html.Tr(
                [
                    dcc.Checklist(
                        [],
                        inline=True,
                        id="checklist-save-additional",
                    ),
                ]
            ),
            html.Tr(html.Td(dbc.Button("Save Particles", id="button-save"))),
            dcc.Download(id="download-file"),
        ],
        style={"overflow": "hidden", "margin": "3px", "width": "100%"},
    )

    cleaning_params_card = collapsing_card(
        card("Cleaning", param_table),
        "clean",
    )
    upload_card = collapsing_card(
        card("Choose File", upload_table), "upload", start_open=True
    )
    graph_controls_card = collapsing_card(
        card("Graph Controls", graph_controls_table), "graph-control"
    )

    save_card = collapsing_card(card("Save Result", save_table), "save")

    graph = dcc.Graph(
        id="graph-picking",
        figure=EMPTY_FIG,
        config={
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "custom_image",
                "height": 10000,
                "width": 10000,
            }
        },
    )

    app.layout = html.Div(
        [
            dbc.Row(html.H1("MagpiEM")),
            dbc.Row(
                html.Table(
                    html.Tr(
                        [
                            html.Td(upload_card),
                            html.Td(cleaning_params_card),
                            html.Td(graph_controls_card),
                            html.Td(save_card),
                        ]
                    )
                )
            ),
            dbc.Row(
                html.Td(
                    dbc.Button(
                        "Next Tomogram",
                        id="button-next-Tomogram",
                        size="lg",
                        style={"width": "100%", "height": "100px"},
                    )
                )
            ),
            dbc.Row(
                html.Div(
                    id="div-graph-data",
                    style={
                        "minHeight": "80px",
                        "height": "80px",
                        "overflow": "auto",
                        "padding": "12px",
                        "margin": "8px 0"
                    }
                )
            ),
            dbc.Row(
                [
                    dbc.Progress(
                        value=0,
                        id="progress-processing",
                        animated=True,
                        striped=True,
                        style={"height": "30px"},
                    ),
                    dcc.Interval(id="interval-processing", interval=100),
                    dcc.Interval(
                        id="interval-clear-points", interval=200, disabled=True
                    ),
                ]
            ),
            dbc.Row([graph]),
            html.Div(
                [
                    dcc.Store(id="store-lattice-data"),
                    dcc.Store(id="store-tomogram-data"),
                    dcc.Store(id="store-selected-lattices"),
                    dcc.Store(id="store-clicked-point"),
                    dcc.Store(id="store-camera"),
                    dcc.Store(id="store-last-click", data=0.0),
                ]
            ),
            html.Footer(
                html.Div(
                    dcc.Link(
                        "Documentation and instructions",
                        href="https://github.com/fnight128/MagpiEM",
                        target="_blank",
                    )
                )
            ),
            emptydiv,
            dcc.ConfirmDialog(
                id="confirm-cant-save-progress",
                message="Progress can only be saved if cleaning was run on all tomograms.",
            ),
        ],
    )

    @app.callback(
        Output("download-file", "data"),
        State("input-save-filename", "value"),
        State("upload-data", "filename"),
        State("switch-keep-particles", "on"),
        State("checklist-save-additional", "value"),
        Input("button-save", "n_clicks"),
        prevent_initial_call=True,
        long_callback=True,
    )
    def save_result(output_name, input_name, keep_selected, save_additional, _):
        global __dash_tomograms
        raise NotImplementedError(
            "save_result function still uses __dash_tomograms and needs to be refactored to use store-tomogram-data"
        )
        if not output_name:
            return None
        if output_name == input_name:
            print("Output and input file cannot be identical")
            return None

        saving_ids = {
            tomo_name: tomo.selected_particle_ids(keep_selected)
            for tomo_name, tomo in __dash_tomograms.items()
        }

        #  temporarily disabled until em file saving is fixed
        #
        # if ".em (Place Object)" in save_additional:
        #     write_emfile(tomograms, "out", keep_selected)
        #     zip_files(output_name, "em")
        #     dcc.send_file(TEMP_FILE_DIR + output_name)

        if ".mat" in input_name:
            write_emc_mat(
                saving_ids,
                TEMP_FILE_DIR + output_name,
                TEMP_FILE_DIR + input_name,
            )
        elif ".star" in input_name:
            write_relion_star(
                saving_ids,
                TEMP_FILE_DIR + output_name,
                TEMP_FILE_DIR + input_name,
            )
        # files = glob.glob(TEMP_FILE_DIR + "*")
        # print(files)
        # print(download(output_name))
        out_file = TEMP_FILE_DIR + output_name
        print(out_file)
        return dcc.send_file(out_file)

    if open_browser:
        webbrowser.open("http://localhost:8050/")
    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
