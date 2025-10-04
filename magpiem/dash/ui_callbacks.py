# -*- coding: utf-8 -*-
"""
UI state management and navigation callbacks for MagpiEM.
"""

import logging
from dash import State, ctx
from dash_extensions.enrich import Input, Output

logger = logging.getLogger(__name__)


def register_ui_callbacks(app, temp_file_dir):
    """Register UI state management and navigation callbacks."""

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
        Output("div-flip-particles", "style"),
        Input("switch-allow-flips", "on"),
        prevent_initial_call=True,
    )
    def update_flipping_switch(flipping_enabled):
        if flipping_enabled:
            return {"visibility": "visible"}
        else:
            return {"visibility": "hidden"}

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
