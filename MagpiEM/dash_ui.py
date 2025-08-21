# -*- coding: utf-8 -*-
"""
Main Dash application for MagpiEM.
"""

import os
import webbrowser

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from flask import Flask

from .callbacks import register_callbacks
from .cache import get_cache_functions
from .layout import create_main_layout

# Constants
TEMP_FILE_DIR = "static/"


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

    # Set up the layout
    app.layout = create_main_layout()

    # Get cache functions and register callbacks
    cache_functions = get_cache_functions()
    register_callbacks(app, cache_functions, TEMP_FILE_DIR)

    if open_browser:
        webbrowser.open("http://localhost:8050/")
    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
