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


from .classes import Cleaner
from .read_write import read_relion, read_emc, modify_relion_star, modify_emc_mat


WHITE = "#FFFFFF"
GREY = "#646464"
BLACK = "#000000"

tomograms = dict()

last_click = 0.0


def main():
    server = Flask(__name__)
    app = DashProxy(
        server=server,
        external_stylesheets=[dbc.themes.SOLAR],
        transforms=[MultiplexerTransform()],
    )
    load_figure_template("SOLAR")

    TEMP_FILE_DIR = "static/"

    if not os.path.exists(TEMP_FILE_DIR):
        os.makedirs(TEMP_FILE_DIR)

    # set 'u' invalid so no points will show
    empty_graph = go.Figure(
        data=go.Cone(x=[0], y=[0], z=[0], u=[[0]], v=[0], w=[0], showscale=False)
    )

    tomograms = dict()

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

    def colour_range(num_points):
        hsv_tuples = [(x * 1.0 / num_points, 0.75, 0.75) for x in range(num_points)]
        rgb_tuples = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
        return [
            "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
            for (r, g, b) in rgb_tuples
        ]

    # def mesh3d_trace(df, colour, opacity):
    #     return go.Mesh3d(
    #         x=df["x"], y=df["y"], z=df["z"], opacity=opacity, color=colour, alphahull=-1
    #     )

    def scatter3d_trace(df, colour, opacity):
        return go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            text=df["n"],
            marker=dict(size=6, color=colour, opacity=opacity),
            showlegend=False,
        )

    def cone_trace(df, colour, opacity, sizeref=10.0):
        return go.Cone(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            u=df["u"],
            v=df["v"],
            w=df["w"],
            text=df["n"],
            sizemode="scaled",
            sizeref=sizeref,
            colorscale=[[0, colour], [1, colour]],
            showscale=False,
            opacity=opacity,
        )

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
        global tomograms
        global last_click

        params_message = ""

        # must always return a graph object or breaks dash
        if not tomo_selection or tomo_selection not in tomograms.keys():
            return empty_graph, params_message

        tomo = tomograms[tomo_selection]

        # prevent clicked point lingering between callbacks
        if ctx.triggered_id != "graph-picking":
            clicked_point = None
        else:
            # strange error with cone plots makes completely random, erroneous clicks
            # happen right after clicking on cone plot - adding a cooldown
            # prevents this
            if time() - last_click < 0.5:
                raise PreventUpdate
            last_click = time()
            clicked_particle_pos = [
                clicked_point["points"][0][c] for c in ["x", "y", "z"]
            ]
            print("Clicked pos", clicked_particle_pos)
            params_message = tomo.show_particle_data(clicked_particle_pos)

        should_make_cones = make_cones

        has_ref = hasattr(tomo, "reference_df")

        fig = go.Figure()
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
        fig.update_layout(scene_aspectmode="data")
        # keep camera constant
        fig["layout"]["uirevision"] = "a"

        # if tomo not yet cleaned, just plot all points
        if len(tomo.lattices) == 1:
            if should_make_cones:

                nonchecking_df = pd.concat(
                    (tomo.nonchecking_particles_df(), tomo.cone_fix_df())
                )
                checking_df = pd.concat(
                    (tomo.checking_particles_df(), tomo.cone_fix_df())
                )
                fig.add_trace(cone_trace(nonchecking_df, WHITE, 0.6, cone_size))
                fig.add_trace(cone_trace(checking_df, BLACK, 1, cone_size))
            else:
                fig.add_trace(scatter3d_trace(tomo.all_particles_df(), WHITE, 0.6))
                fig.add_trace(scatter3d_trace(tomo.checking_particles_df(), BLACK, 1))

            return fig, params_message

        if clicked_point:
            tomo.toggle_selected(clicked_point["points"][0]["text"])

        array_dict = tomo.particle_df_dict

        # define linear range of colours
        hex_vals = colour_range(len(array_dict))

        # assign one colour to each protein array index
        colour_dict = dict()
        for idx, akey in enumerate(array_dict.keys()):
            hex_val = WHITE if akey in tomo.selected_n else hex_vals[idx]
            colour_dict.update({akey: hex_val})

        # assign colours and plot array
        for akey in array_dict.keys():
            array = array_dict[akey]
            opacity = 1
            if akey == 0:
                if show_removed:
                    colour = BLACK
                    opacity = 0.6
                else:
                    continue
            else:
                colour = colour_dict[akey]

            if should_make_cones:
                # cone fix
                array = pd.concat([tomo.cone_fix_df(), array])
                fig.add_trace(cone_trace(array, colour, opacity, cone_size))
                if has_ref:
                    fig.add_trace(scatter3d_trace(tomo.reference_df, GREY, 0.2))
            else:
                fig.add_trace(scatter3d_trace(array, colour, opacity))
                # fig.add_trace(mesh3d_trace(array, colour, opacity))
                if has_ref:
                    fig.add_trace(scatter3d_trace(tomo.reference_df, GREY, 0.2))
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
        # dummy
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
        global tomograms
        if all_tomos:
            for tomo in tomograms.values():
                tomo.toggle_convex_arrays()
        else:
            tomograms[current_tomo].toggle_convex_arrays()
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
        global tomograms
        if all_tomos:
            for tomo in tomograms.values():
                tomo.toggle_concave_arrays()
        else:
            tomograms[current_tomo].toggle_concave_arrays()
        return int(clicks or 0) + 1

    @app.callback(
        Output("download-file", "data"),
        Input("button-save-progress", "n_clicks"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def save_current_progress(clicks, filename):
        if not clicks:
            return None

        file_path = TEMP_FILE_DIR + filename + "_progress.yml"

        cleaning_params_key = ".__cleaning_parameters__."
        global tomograms
        tomo_dict = {}
        for name, tomo in tomograms.items():
            if name == cleaning_params_key:
                print("Tomo {} has an invalid name and cannot be saved!".format(name))
                continue
            tomo_dict[name] = tomo.write_prog_dict()
        try:
            tomo_dict[".__cleaning_parameters__."] = next(
                iter(tomograms.values())
            ).cleaning_params.dict_to_print
        except Exception:
            print("No cleaning parameters found to save")
        print("Saving keys:", tomo_dict.keys())
        prog = yaml.safe_dump(tomo_dict)
        with open(file_path, "w") as yaml_file:
            yaml_file.write(prog)
        return dcc.send_file(file_path)

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
        global tomograms

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

        tomograms = read_uploaded_tomo(data_path)
        if not tomograms:
            return "Particle database (.mat/.star) unreadable", *failed_upload

        try:
            with open(prev_path, "r") as prev_yaml:
                prev_yaml = yaml.safe_load(prev_yaml)
        except Exception:
            return "Previous session file unreadable", *failed_upload

        # check keys line up between files
        geom_keys = set(tomograms.keys())
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

        for tomo_name, tomo in tomograms.items():
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
        global tomograms

        # unfortunately need to merge two callbacks here, dash does not allow multiple
        # callbacks with the same output so use ctx to distinguish between cases
        try:
            tomo_keys = list(tomograms.keys())
            tomo_key_0 = tomo_keys[0]
        except Exception:
            return [], ""

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
            tomograms = read_emc(data_path, num_images=num_images)
        elif ".star" in data_path:
            tomograms = read_relion(data_path, num_images=num_images)
        else:
            return
        return tomograms

    @app.callback(
        Output("label-read", "children"),
        Output("dropdown-tomo", "disabled"),
        Output("collapse-upload", "is_open"),
        Output("collapse-graph-control", "is_open"),
        Output("collapse-clean", "is_open"),
        Output("collapse-proximity", "is_open"),
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
            return "", True, True, False, False, False

        num_img_dict = {0: 1, 1: 5, 2: -1}
        num_images = num_img_dict[num_images]

        global tomograms

        # ensure temp directory clear
        files = glob.glob(TEMP_FILE_DIR + "*")
        print(files)
        for f in files:
            os.remove(f)

        save_dash_upload(filename, contents)
        temp_file_path = TEMP_FILE_DIR + filename

        tomograms = read_uploaded_tomo(temp_file_path, num_images)

        if not tomograms:
            return "File Unreadable", True, True, False, False, False
        return "Tomograms read", False, False, True, True, False

    def clean_tomo(tomo, clean_params):
        tomo.set_clean_params(clean_params)

        tomo.reset_cleaning()

        tomo.autoclean()

        tomo.generate_lattice_dfs()

    def prox_clean_tomo(tomo, dist_min, dist_max):
        if not any(tomo.reference_points):
            print("tomo {} has no reference points uploaded".format(tomo.name))
        t0 = time()

        tomo.proximity_clean((dist_min**2, dist_max**2))

        print("time for {}:".format(tomo.name), time() - t0)

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

        global tomograms

        if ctx.triggered_id == "button-preview-clean":
            print("Preview")
            clean_tomo(list(tomograms.values())[0], clean_params)
            return False, True, True
        else:
            print("Full")
            clean_count = 0
            clean_total_time = 0
            total_tomos = len(tomograms.keys())
            for tomo in tomograms.values():
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

    prox_table = html.Table(
        [
            html.Tr(
                [
                    html.Td(
                        dcc.Upload(
                            id="upload-ref",
                            children="Choose Reference",
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
                            multiple=True,
                        ),
                        colSpan=4,
                    )
                ]
            ),
            html.Tr(
                [
                    html.Td("Files requiring references: "),
                    html.Td(
                        html.Div(id="div-need-refs", className="text-danger"), colSpan=3
                    ),
                ]
            ),
            html.Tr(
                [
                    html.Td("Distance from Reference Particles"),
                    inp_num("dist-min", 30),
                    html.Td("-"),
                    inp_num("dist-max", 35),
                ]
            ),
            html.Tr(
                html.Td(
                    dbc.Button(
                        "Preview Cleaning", id="button-preview-prox", color="secondary"
                    ),
                    colSpan=4,
                ),
            ),
            html.Tr(
                html.Td(
                    dbc.Button("Run Full Cleaning", id="button-full-prox"), colSpan=4
                ),
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
            html.Tr(
                # dcc.RadioItems(
                #     [
                #        "Clean based on orientation",
                #         # "Clean based on reference particles",
                #     ],
                #     "Clean based on orientation",
                #     id="radio-cleantype",
                # )
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

    cleaning_params_card = collapsing_card(card("Cleaning", param_table), "clean")
    proximity_params_card = collapsing_card(
        card("Proximity Cleaning", prox_table), "proximity"
    )
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
                            html.Td(proximity_params_card),
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
            emptydiv,
        ]
    )

    @app.callback(
        Output("graph-picking", "style"),
        Input("dropdown-tomo", "value"),
    )
    def graph_visibility(t_id):
        global tomograms
        if not t_id:
            return {"display": "none"}
        elif t_id not in tomograms.keys():
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
    def save_result(output_name, input_name, keep_selected, save_additional, clicks):
        global tomograms
        if not output_name:
            return None
        if output_name == input_name:
            print("Output and input file cannot be identical")
            return None

        saving_ids = {
            tomo_name: tomo.selected_particle_ids(keep_selected)
            for tomo_name, tomo in tomograms.items()
        }

        #  temporarily disabled until em file saving is fixed
        #
        # if ".em (Place Object)" in save_additional:
        #     write_emfile(tomograms, "out", keep_selected)
        #     zip_files(output_name, "em")
        #     dcc.send_file(TEMP_FILE_DIR + output_name)

        if ".mat" in input_name:
            modify_emc_mat(
                saving_ids,
                TEMP_FILE_DIR + output_name,
                TEMP_FILE_DIR + input_name,
            )
        elif ".star" in input_name:
            modify_relion_star(
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
