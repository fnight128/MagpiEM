# -*- coding: utf-8 -*-
"""
Main Dash application for MagpiEM.
"""

import argparse
import logging
import os
import sys
import webbrowser

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from flask import Flask
import diskcache
from dash.long_callback import DiskcacheLongCallbackManager

from .callbacks import register_callbacks
from .plot_cache import get_cache_functions
from .layout import create_main_layout
from .utilities import setup_cleanup_handlers

# Constants
TEMP_FILE_DIR = "cache/"
CLEANING_PARAMS_DIR = "past_cleaning_params/"


def configure_logging(level=logging.WARNING):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def main(open_browser=True, log_level=logging.WARNING):
    """
    Main function to run the MagpiEM Dash application.

    Parameters
    ----------
    open_browser : bool, optional
        Whether to automatically open the application in the default browser.
        Default is True. Set to False for testing or headless operation.
    log_level : int, optional
        Logging level to use. Default is WARNING.
    """
    try:
        args = parse_arguments()
        open_browser = not args.no_browser
        log_level = getattr(logging, args.log_level.upper())
    except SystemExit:
        # Help was requested or invalid arguments
        return
    server = Flask(__name__)

    # Set up diskcache for background callbacks
    cache = diskcache.Cache(f"./{TEMP_FILE_DIR}")
    long_callback_manager = DiskcacheLongCallbackManager(cache)

    app = DashProxy(
        server=server,
        external_stylesheets=[dbc.themes.SOLAR],
        transforms=[MultiplexerTransform()],
        title="MagpiEM",
        update_title=None,  # Prevent "Updating..." title
        long_callback_manager=long_callback_manager,
    )
    load_figure_template("SOLAR")

    # Configure logging
    configure_logging(log_level)

    if not os.path.exists(TEMP_FILE_DIR):
        os.makedirs(TEMP_FILE_DIR)

    if not os.path.exists(CLEANING_PARAMS_DIR):
        os.makedirs(CLEANING_PARAMS_DIR)

    # Set up cleanup handlers to clear cache on termination
    setup_cleanup_handlers(TEMP_FILE_DIR)

    # Set up the layout
    app.layout = create_main_layout()

    # Get cache functions and register callbacks
    cache_functions = get_cache_functions()
    register_callbacks(app, cache_functions, TEMP_FILE_DIR, CLEANING_PARAMS_DIR)

    if open_browser:
        webbrowser.open("http://localhost:8050/")
    app.run_server(debug=True, use_reloader=False)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MagpiEM - Particle Cleaning Application"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open the browser"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
