# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:53:45 2022

@author: Frank
"""
import colorsys
import pandas as pd
import plotly.graph_objects as go
import os
import base64
import math
import yaml
import json
from pathlib import Path

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

from .classes import Cleaner, simple_figure
from .read_write import read_relion_star, read_emc_mat, write_relion_star, write_emc_mat

WHITE = "#FFFFFF"
GREY = "#646464"
BLACK = "#000000"

__dash_tomograms = dict()
EMPTY_FIG = simple_figure()

__last_click = 0.0
__progress = 0.0

TEMP_FILE_DIR = "static/"
__CLEAN_YAML_NAME = "prev_clean_params.yml"


def main():
    server = Flask(__name__)
    app = DashProxy(
        server=server,
        external_stylesheets=[dbc.themes.SOLAR],
        transforms=[MultiplexerTransform()],
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

    @app.callback(
        Output("graph-picking", "figure"),
        Output("div-graph-data", "children"),
        Input("dropdown-tomo", "value"),
        Input("graph-picking", "clickData"),
        Input("switch-cone-plot", "on"),
        State("inp-cone-size", "value"),
        Input("button-set-cone-size", "n_clicks"),
        Input("switch-show-removed", "on"),
        Input("button-next-Tomogram", "disabled"),
        prevent_initial_call=True,
    )
    def plot_tomo(
        tomo_selection: str,
        clicked_point,
        make_cones: bool,
        cone_size: float,
        _,
        show_removed: bool,
        __,
    ):
        global __dash_tomograms
        global __last_click

        if not make_cones:
            cone_size = -1

        params_message = ""

        # must always return a graph object or can break dash
        if not tomo_selection or tomo_selection not in __dash_tomograms.keys():
            return EMPTY_FIG, params_message

        tomo = __dash_tomograms[tomo_selection]

        # prevent clicked point lingering between callbacks
        if ctx.triggered_id != "graph-picking":
            clicked_point = None
        else:
            # strange error with cone plots makes completely random, erroneous clicks
            # happen right after clicking on cone plot - adding a cooldown
            # prevents this
            if time() - __last_click < 0.5:
                raise PreventUpdate
            __last_click = time()
            clicked_particle_pos = [
                clicked_point["points"][0][c] for c in ["x", "y", "z"]
            ]
            params_message = tomo.show_particle_data(clicked_particle_pos)
            tomo.toggle_selected(clicked_point["points"][0]["text"])

        fig = tomo.plot_all_lattices(
            cone_size=cone_size, showing_removed_particles=show_removed
        )

        return fig, params_message

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
        prevent_initial_callback=True,
    )
    def open_cards(_):
        upload_phase = True, False, False, False
        cleaning_phase = False, True, True, False
        saving_phase = False, False, True, True
        global __dash_tomograms

        if not __dash_tomograms:
            return upload_phase

        random_tomo = next(iter(__dash_tomograms.values()))
        if len(random_tomo.lattices.keys()) > 1:
            return saving_phase
        else:
            return cleaning_phase

    @app.callback(
        Output("dropdown-tomo", "options"),
        Output("dropdown-tomo", "value"),
        State("dropdown-tomo", "value"),
        Input("dropdown-tomo", "disabled"),
        Input("button-next-Tomogram", "n_clicks"),
        Input("button-previous-Tomogram", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_dropdown(current_val, disabled, _, __):
        global __dash_tomograms

        tomo_keys = list(__dash_tomograms.keys())

        # unfortunately need to merge two callbacks here, dash does not allow multiple
        # callbacks with the same output so use ctx to distinguish between cases
        if len(tomo_keys) == 0:
            return [], ""

        tomo_key_0 = tomo_keys[0]

        # enabling dropdown once cleaning finishes
        if ctx.triggered_id == "dropdown-tomo":
            return tomo_keys, tomo_key_0

        # moving to next/prev item in dropdown when next/prev Tomogram button pressed
        if not current_val:
            return tomo_keys, ""

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
            tomograms = read_emc_mat(data_path, num_images=num_images)
        elif ".star" in data_path:
            tomograms = read_relion_star(data_path, num_images=num_images)
        else:
            return
        return tomograms

    def read_previous_progress(progress_file):
        global __dash_tomograms

        try:
            with open(progress_file, "r") as prev_yaml:
                prev_yaml = yaml.safe_load(prev_yaml)
        except yaml.YAMLError:
            return "Previous session file unreadable"

        # check keys line up between files
        geom_keys = set(__dash_tomograms.keys())
        prev_keys = set(prev_yaml.keys())
        prev_keys.discard(".__cleaning_parameters__.")

        print(geom_keys)
        print(prev_keys)

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

        for tomo_name, tomo in __dash_tomograms.items():
            tomo.apply_progress_dict(prev_yaml[tomo_name])

    @app.callback(
        Output("label-read", "children"),
        Output("dropdown-tomo", "disabled"),
        Output("store-tomograms", "data"),
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
            return "Please choose a particle database", True

        global __progress
        __progress = 0.0

        num_img_dict = {0: 1, 1: 5, 2: -1}
        num_images = num_img_dict[num_images]

        global __dash_tomograms

        # ensure temp directory clear
        all_files = glob.glob(TEMP_FILE_DIR + "*")
        all_files = [file for file in all_files if __CLEAN_YAML_NAME not in file]
        if all_files:
            print("Pre-existing temp files found, removing:", all_files)
            for f in all_files:
                os.remove(f)

        save_dash_upload(filename, contents)

        data_file_path = TEMP_FILE_DIR + filename

        __dash_tomograms = read_uploaded_tomo(
            data_file_path, __progress, num_images=num_images
        )

        if not __dash_tomograms:
            return "Data file Unreadable", True

        if ctx.triggered_id == "upload-previous-session":
            save_dash_upload(previous_filename, previous_contents)
            progress_path = TEMP_FILE_DIR + previous_filename
            read_previous_progress(progress_path)

        store_tomograms = {
            tomogram[0]: tomogram[1].to_dict() for tomogram in __dash_tomograms.items()
        }

        with open("demofile.txt", "w") as f:
            json.dump(store_tomograms, f)

        return "Tomograms read", False, {}

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
    def read_previous_clean_params(is_open):
        global TEMP_FILE_DIR, __CLEAN_YAML_NAME
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
            with open(TEMP_FILE_DIR + __CLEAN_YAML_NAME, "r") as prev_yaml:
                prev_yaml = yaml.safe_load(prev_yaml)
                prev_vals = [prev_yaml[key] for key in clean_keys]
                return prev_vals
        except FileNotFoundError or yaml.YAMLError or KeyError:
            print("Couldn't find or read a previous cleaning file.")
            raise PreventUpdate

    def clean_tomo(tomo, clean_params):
        tomo.set_clean_params(clean_params)

        tomo.reset_cleaning()

        tomo.autoclean()

        tomo.generate_lattice_dfs()

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
        clicks,
        clicks2,
    ):
        if not clicks or clicks2:
            return True, True, False
        # print(inp_file)
        # tomo = Tomogram(name, particles)

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

        global __dash_tomograms, __clean_yaml_name, __progress

        __progress = 0.0

        if ctx.triggered_id == "button-preview-clean":
            print("Preview")
            clean_tomo(list(__dash_tomograms.values())[0], clean_params)
            return False, True, True
        else:
            print("Full")
            clean_count = 0
            clean_total_time = 0
            total_tomos = len(__dash_tomograms.keys())
            for tomo in __dash_tomograms.values():
                t0 = time()
                clean_tomo(tomo, clean_params)
                clean_total_time += time() - t0
                clean_count += 1
                tomos_remaining = total_tomos - clean_count
                clean_speed = clean_total_time / clean_count
                secs_remaining = clean_speed * tomos_remaining
                formatted_time_remaining = str(
                    datetime.timedelta(seconds=secs_remaining)
                ).split(".")[0]
                __progress = clean_count / total_tomos
                print("Time remaining:", formatted_time_remaining)
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
        with open(TEMP_FILE_DIR + __CLEAN_YAML_NAME, "w") as yaml_file:
            yaml_file.write(yaml.safe_dump(cleaning_params_dict))
        return False, False, True

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
            dbc.Row(html.Div(id="div-graph-data")),
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
                ]
            ),
            dbc.Row([graph]),
            html.Div(dcc.Store(id="store-tomograms", data={})),
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

    webbrowser.open("http://localhost:8050/")
    app.run_server(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
