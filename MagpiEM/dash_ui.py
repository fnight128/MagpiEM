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

import MagpiEM.read_write
from .classes import Cleaner
from .read_write import read_relion_star, read_emc_mat, write_relion_star, write_emc_mat

WHITE = "#FFFFFF"
GREY = "#646464"
BLACK = "#000000"

__dash_tomograms = dict()

__last_click = 0.0

TEMP_FILE_DIR = "static/"


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

    # set 'u' invalid so no points will show
    empty_graph = go.Figure(
        data=go.Cone(x=[0], y=[0], z=[0], u=[[0]], v=[0], w=[0], showscale=False)
    )

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

    def prog_bar(current_index, goal):
        if current_index < 0 or goal < 0 or current_index > goal:
            return "Unable to generate progress bar"
        bar_length = 20
        progress = math.floor((current_index / goal) * bar_length)
        to_do = bar_length - progress
        bar = "<{}{}> {}/{}".format("#" * progress, "-" * to_do, current_index, goal)
        return bar

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

        # must always return a graph object or breaks dash
        if not tomo_selection or tomo_selection not in __dash_tomograms.keys():
            return empty_graph, params_message

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
            print("Clicked pos", clicked_particle_pos)
            params_message = tomo.show_particle_data(clicked_particle_pos)
            tomo.toggle_selected(clicked_point["points"][0]["text"])

        fig = tomo.plot_all_lattices(
            showing_removed_particles=show_removed, cone_size=cone_size
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

        cleaning_params_key = ".__cleaning_parameters__."
        global __dash_tomograms
        tomo_dict = {}
        for name, tomo in __dash_tomograms.items():
            if name == cleaning_params_key:
                print("Tomo {} has an invalid name and cannot be saved!".format(name))
                continue
            tomo_dict[name] = tomo.write_prog_dict()
        try:
            tomo_dict[".__cleaning_parameters__."] = next(
                iter(__dash_tomograms.values())
            ).cleaning_params.dict_to_print
        except Exception:
            print("No cleaning parameters found to save")
        print("Saving keys:", tomo_dict.keys())
        prog = yaml.safe_dump(tomo_dict)
        with open(file_path, "w") as yaml_file:
            yaml_file.write(prog)
        return dcc.send_file(file_path), False

    @app.callback(
        Output("label-read", "children"),
        Output("dropdown-tomo", "disabled"),
        Output("collapse-upload", "is_open"),
        Output("collapse-graph-control", "is_open"),
        Output("collapse-save", "is_open"),
        Input("upload-previous-session", "filename"),
        Input("upload-previous-session", "contents"),
        State("upload-data", "filename"),
        State("upload-data", "contents"),
        prevent_initial_call=True,
    )
    def load_previous_progress(
            previous_filename, previous_contents, data_filename, data_contents
    ):
        global __dash_tomograms

        failed_upload = [True, True, False, False]
        successful_upload = [False, False, True, True]
        if ctx.triggered_id != "upload-previous-session":
            data_filename = None

        if not previous_filename:
            return "", *failed_upload
        if not data_filename:
            return (
                "Please select a particle database (.mat or .star) first",
                *failed_upload,
            )

        # ensure temp directory clear
        files = glob.glob(TEMP_FILE_DIR + "*")
        if files:
            print("Pre-existing temp files found, removing:", files)
        for f in files:
            os.remove(f)

        save_dash_upload(previous_filename, previous_contents)
        save_dash_upload(data_filename, data_contents)

        data_path = TEMP_FILE_DIR + data_filename
        prev_path = TEMP_FILE_DIR + previous_filename

        __dash_tomograms = read_uploaded_tomo(data_path)
        if not __dash_tomograms:
            return "Particle database (.mat/.star) unreadable", *failed_upload

        try:
            with open(prev_path, "r") as prev_yaml:
                prev_yaml = yaml.safe_load(prev_yaml)
        except yaml.YAMLError:
            return "Previous session file unreadable", *failed_upload

        # check keys line up between files
        geom_keys = set(__dash_tomograms.keys())
        prev_keys = set(prev_yaml.keys())
        prev_keys.discard(".__cleaning_parameters__.")

        print("geom keys", geom_keys)
        print("prev keys", prev_keys)

        if not geom_keys == prev_keys:
            if len(prev_keys) in {1, 5}:
                # likely saved result after only loading few tomograms
                return [
                    "Keys do not match up between previous session and .mat file.",
                    html.Br(),
                    "Previous session only contains {0} keys, did you save the session with only {0} Tomogram(s) "
                    "loaded?".format(len(prev_keys)),
                ], *failed_upload
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
            ], *failed_upload

        for tomo_name, tomo in __dash_tomograms.items():
            tomo.apply_prog_dict(prev_yaml[tomo_name])

        return "", *successful_upload

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

    def read_uploaded_tomo(data_path, num_images=-1):
        if ".mat" in data_path:
            tomograms = read_emc_mat(data_path, num_images=num_images)
        elif ".star" in data_path:
            tomograms = read_relion_star(data_path, num_images=num_images)
        else:
            return
        return tomograms

    @app.callback(
        Output("label-read", "children"),
        Output("dropdown-tomo", "disabled"),
        Output("collapse-upload", "is_open"),
        Output("collapse-graph-control", "is_open"),
        Output("collapse-clean", "is_open"),
        Input("button-read", "n_clicks"),
        State("upload-data", "filename"),
        State("upload-data", "contents"),
        State("slider-num-images", "value"),
        long_callback=True,
        prevent_initial_call=True,
    )
    def read_tomograms(clicks, filename, contents, num_images):
        if ctx.triggered_id != "button-read":
            filename = None

        if not filename:
            return "", True, True, False, False

        num_img_dict = {0: 1, 1: 5, 2: -1}
        num_images = num_img_dict[num_images]

        global __dash_tomograms

        # ensure temp directory clear
        files = glob.glob(TEMP_FILE_DIR + "*")
        print(files)
        for f in files:
            os.remove(f)

        save_dash_upload(filename, contents)
        temp_file_path = TEMP_FILE_DIR + filename

        __dash_tomograms = read_uploaded_tomo(temp_file_path, num_images)

        if not __dash_tomograms:
            return "File Unreadable", True, True, False, False
        return "Tomograms read", False, False, True, True

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
            disp_goal: float,
            disp_tol: float,
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

        clean_params = Cleaner(
            cc_thresh,
            min_neighbours,
            array_size,
            dist_goal,
            dist_tol,
            ori_goal,
            ori_tol,
            disp_goal,
            disp_tol,
            allow_flips,
        )

        print("Clean")

        global __dash_tomograms

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
                print(prog_bar(clean_count, total_tomos))
                print("Time remaining:", formatted_time_remaining)
                print()
        return False, False, True

    size = "50px"

    def inp_num(id_suffix, default=None):
        return html.Td(
            dbc.Input(
                id="inp-" + id_suffix,
                type="number",
                value=default,
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
                    inp_num("dist-goal", 25),
                    html.Td("±"),
                    inp_num("dist-tol", 10),
                ]
            ),
            html.Tr(
                [
                    html.Td("Orientation (°)"),
                    inp_num("ori-goal", 9),
                    html.Td("±"),
                    inp_num("ori-tol", 10),
                ]
            ),
            html.Tr(
                [
                    html.Td("Curvature (°)"),
                    inp_num("curv-goal", 90),
                    html.Td("±"),
                    inp_num("curv-tol", 20),
                ]
            ),
            html.Tr([html.Td("Min. Neighbours"), inp_num("min-neighbours", 2)]),
            html.Tr([html.Td("CC Threshold"), inp_num("cc-thresh", 5)]),
            html.Tr([html.Td("Min. Lattice Size"), inp_num("array-size", 5)]),
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
                "height": "100%"
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
            html.Tr(
                [
                    html.Td(dbc.Button("Read tomograms", id="button-read")),
                ]
            ),
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
                    ),
                ]
            ),
            html.Tr(
                [
                    html.Td("Cone Plot (experimental!)", id="label-cone-plot"),
                    daq.BooleanSwitch(id="switch-cone-plot", on=False),
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

    cleaning_params_card = collapsing_card(card("Cleaning", param_table), "clean", )
    upload_card = collapsing_card(
        card("Choose File", upload_table), "upload", start_open=True
    )
    graph_controls_card = collapsing_card(
        card("Graph Controls", graph_controls_table), "graph-control"
    )

    save_card = collapsing_card(card("Save Result", save_table), "save")

    graph = dcc.Graph(id="graph-picking", figure=empty_graph)

    emptydiv = html.Div(
        [
            # nonexistent outputs for callbacks with no visible effect
            html.Div(id="div-null", style={"display": "none"}),
        ],
        style={"display": "none"},
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
            dbc.Row([graph]),
            dbc.Row(html.Div(dcc.Link("Documentation and instructions", href="https://github.com/fnight128/MagpiEM",
                                      target="_blank"))),
            emptydiv,
            dcc.ConfirmDialog(
                id='confirm-cant-save-progress',
                message='Progress can only be saved if cleaning was run on all tomograms.',
            ),
        ],
    )

    @app.callback(
        Output("graph-picking", "style"),
        Input("dropdown-tomo", "value"),
    )
    def graph_visibility(t_id):
        global __dash_tomograms
        if not t_id:
            return {"display": "none"}
        elif t_id not in __dash_tomograms.keys():
            return {"display": "none"}
        else:
            return {}

    @app.callback(
        # Output("link-download", "href"),
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
