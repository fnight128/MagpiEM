# -*- coding: utf-8 -*-
"""
Upload and session management callbacks for MagpiEM.
"""

import base64
import logging
import os
import uuid
from pathlib import Path

import yaml
from dash import State, ctx, html
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output

from .io_utils import (
    get_tomogram_names,
    process_uploaded_file,
    load_previous_session,
)

logger = logging.getLogger(__name__)


def register_upload_callbacks(app, temp_file_dir, cleaning_params_dir):
    """Register upload and session management callbacks."""

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
        Output("button-read", "disabled"),
        Input("upload-previous-session", "filename"),
        prevent_initial_call=True,
    )
    def hide_read_button(_):
        return True

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
                f"Couldn't find or read a previous cleaning file for {filename} "
                f"at {cleaning_params_dir}."
            )
            raise PreventUpdate

    def save_dash_upload(filename, contents, temp_file_dir):
        logger.info("Uploading file: %s", filename)
        data = contents.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(temp_file_dir, filename), "wb") as fp:
            fp.write(base64.decodebytes(data))

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
                    "Previous session only contains {0} keys, "
                    "did you save the session with only {0} Tomogram(s) "
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
