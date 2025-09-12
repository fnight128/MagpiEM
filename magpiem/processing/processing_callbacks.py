# -*- coding: utf-8 -*-
"""
Data processing and cleaning callbacks for MagpiEM.
"""

import logging
from dash import State, dcc
from dash_extensions.enrich import Input, Output

from .classes.cleaner import Cleaner, save_cleaning_parameters
from .processing_utils import process_single_tomogram
from .cpp_integration import clean_tomo_with_cpp, check_cpp_availability
from ..io.io_utils import read_emc_mat, read_emc_tomogram_raw_data
from ..plotting.plot_cache import clear_cache

logger = logging.getLogger(__name__)


def register_processing_callbacks(app, temp_file_dir, cleaning_params_dir):
    """Register data processing and cleaning callbacks."""

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
