# -*- coding: utf-8 -*-
"""
Main callback registration for MagpiEM

Imports and registers all callbacks
"""

from ..io.upload_callbacks import register_upload_callbacks
from ..plotting.plotting_callbacks import register_plotting_callbacks
from ..processing.processing_callbacks import register_processing_callbacks
from .ui_callbacks import register_ui_callbacks
from ..io.save_callbacks import register_save_callbacks


def register_callbacks(app, cache_functions, temp_file_dir, cleaning_params_dir):
    """
    Register all callbacks

    Main entry point for callback registration

    Parameters
    ----------
    app : dash.Dash
        Dash application instance
    cache_functions : dict
        Dictionary containing cache-related functions
    temp_file_dir : str
        Path to temporary file directory
    cleaning_params_dir : str
        Path to cleaning parameters directory
    """
    register_upload_callbacks(app, temp_file_dir, cleaning_params_dir)
    register_plotting_callbacks(app, cache_functions, temp_file_dir)
    register_processing_callbacks(app, temp_file_dir, cleaning_params_dir)
    register_ui_callbacks(app, temp_file_dir)
    register_save_callbacks(app, temp_file_dir)
