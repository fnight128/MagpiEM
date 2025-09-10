# -*- coding: utf-8 -*-
"""
File saving and export callbacks for MagpiEM.
"""

import logging
from dash import State, dcc
from dash_extensions.enrich import Input, Output

from .io_utils import (
    write_emc_mat,
    write_relion_star,
    validate_save_inputs,
    extract_particle_ids_for_saving,
    save_file_by_type,
)
from .mat_file_validation import validate_mat_files

logger = logging.getLogger(__name__)


def register_save_callbacks(app, temp_file_dir):
    """Register file saving and export callbacks."""

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
