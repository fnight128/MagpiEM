# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:53:45 2022

@author: Frank
"""
import colorsys
import pandas as pd
import scipy.io
import plotly.graph_objects as go
import os
import base64
import math
from pathlib import Path

import webbrowser

import glob
from time import time

# import dash
from dash import dcc, html, State, ctx
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
import dash_daq as daq

# flask server to save files
from flask import Flask  # , send_from_directory


from classes import SubTomogram, Cleaner
from read_write import read_imod, write_emfile, modify_emc_mat, zip_files

WHITE = "#FFFFFF"
GREY = "#646464"
BLACK = "#000000"

subtomograms = dict()

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

    subtomograms = dict()

    @app.callback(
        Output("upload-data", "children"),
        Output("input-save-filename", "value"),
        # Output("card-graph-control"),
        Input("upload-data", "filename"),
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
        HSV_tuples = [(x * 1.0 / num_points, 0.75, 0.75) for x in range(num_points)]
        RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
        return [
            "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
            for (r, g, b) in RGB_tuples
        ]

    def mesh3d_trace(df, colour, opacity):
        return go.Mesh3d(
            x=df["x"], y=df["y"], z=df["z"], opacity=opacity, color=colour, alphahull=-1
        )

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

    def cone_trace(df, colour, opacity, sizeref=10):
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
        Input("button-next-subtomogram", "disabled"),
        prevent_initial_call=True,
    )
    def plot_tomo(
        tomo_selection: str, clicked_point, make_cones: bool, cone_size: float, _, show_removed: bool, __
    ):
        global subtomograms
        global last_click
        # print("plot_tomo")

        params_message = ""

        # Warning: must return a graph object in both of these, or breaks dash
        if not tomo_selection or not tomo_selection in subtomograms.keys():
            return empty_graph, params_message

        subtomo = subtomograms[tomo_selection]

        # strange error with cone plots makes completely random, erroneous clicks
        # happen right after clicking on cone plot - add a cooldown to temporarily
        # prevent this - clicks must now be 500ms apart
        # print("Time since last click: ", time() - last_click)
        click_on_cooldown = time() - last_click < 0.5
        # clicked point lingers between calls, causing unwanted toggling when e.g.
        # switching to cones, and selected points can carry over to the next graph
        # prevent by clearing clicked_point if not actually from clicking a point
        if ctx.triggered_id != "graph-picking":
            clicked_point = ""
        else:
            if click_on_cooldown:
                raise PreventUpdate
            last_click = time()
            clicked_particle_pos = [
                clicked_point["points"][0][c] for c in ["x", "y", "z"]
            ]
            print("Clicked pos", clicked_particle_pos)
            params_message = subtomo.show_particle_data(clicked_particle_pos)

        should_make_cones = make_cones and not subtomo.position_only

        try:
            subtomo.reference_df
            has_ref = True
        except:
            has_ref = False

        fig = go.Figure()
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
        fig["layout"]["uirevision"] = "a"

        # if subtomo not yet cleaned, just plot all points
        if len(subtomo.auto_cleaned_particles) == 0:
            if should_make_cones:
                # me from the future: I have no idea what this was for but
                # seems bad, causes weird cone behaviour when checking params
                # disabling for now
                # have to fix dash cones error again
                #scale = 10 if params_message else 1

                nonchecking_df = pd.concat(
                    (subtomo.nonchecking_particles_df(), subtomo.cone_fix_df())
                )
                checking_df = pd.concat(
                    (subtomo.checking_particles_df(), subtomo.cone_fix_df())
                )
                fig.add_trace(cone_trace(nonchecking_df, WHITE, 0.6, cone_size))
                fig.add_trace(cone_trace(checking_df, BLACK, 1, cone_size))
            else:
                fig.add_trace(scatter3d_trace(subtomo.all_particles_df(), WHITE, 0.6))
                fig.add_trace(
                    scatter3d_trace(subtomo.checking_particles_df(), BLACK, 1)
                )

            return fig, params_message

        if clicked_point:
            subtomo.toggle_selected(clicked_point["points"][0]["text"])

        array_dict = subtomo.particle_df_dict

        # define linear range of colours
        hex_vals = colour_range(len(array_dict))

        # assign one colour to each protein array index
        colour_dict = dict()
        for idx, akey in enumerate(array_dict.keys()):
            hex_val = WHITE if akey in subtomo.selected_n else hex_vals[idx]
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
                array = pd.concat([subtomo.cone_fix_df(), array])
                fig.add_trace(cone_trace(array, colour, opacity, cone_size))
                if has_ref:
                    fig.add_trace(scatter3d_trace(subtomo.reference_df, GREY, 0.2))
            else:
                fig.add_trace(scatter3d_trace(array, colour, opacity))
                # fig.add_trace(mesh3d_trace(array, colour, opacity))
                if has_ref:
                    fig.add_trace(scatter3d_trace(subtomo.reference_df, GREY, 0.2))
        return fig, params_message

    @app.callback(
        Output("dropdown-filetype", "value"),
        Input("upload-data", "filename"),
    )
    def update_filetype_dropdown(filename):
        if not filename:
            raise PreventUpdate
        x = Path(filename).suffix
        print(x)
        try:
            ext = Path(filename).suffix
        except:
            raise PreventUpdate
        return ext

    @app.callback(
        Output("label-keep-particles", "children"), Input("switch-keep-particles", "on")
    )
    def update_keeping_particles(keeping):
        if keeping:
            return "Keep selected particles"
        else:
            return "Keep unselected particles"

    @app.callback(
        Output("dropdown-tomo", "options"),
        Output("dropdown-tomo", "value"),
        State("dropdown-tomo", "value"),
        Input("dropdown-tomo", "disabled"),
        Input("button-next-subtomogram", "n_clicks"),
        Input("button-previous-subtomogram", "n_clicks"),
    )
    def update_dropdown(current_val, disabled, _, __):
        global subtomograms

        # unfortunately need to merge two callbacks here, dash does not allow multiple
        # callbacks with the same output so use ctx to distinguish between cases
        try:
            subtomo_keys = list(subtomograms.keys())
            subtomo_key_0 = subtomo_keys[0]
        except:
            return [], ""

        # enabling dropdown once cleaning finishes
        if ctx.triggered_id == "dropdown-tomo":
            return subtomo_keys, subtomo_key_0

        # moving to next/prev item in dropdown when next/prev subtomogram button pressed
        if not current_val:
            return subtomo_keys, ""
        current_index = subtomo_keys.index(current_val)

        if ctx.triggered_id == "button-next-subtomogram":
            # loop back to start if at end of list
            if len(subtomo_keys) == current_index + 1:
                new_val = subtomo_key_0
            else:
                new_val = subtomo_keys[subtomo_keys.index(current_val) + 1]
        elif ctx.triggered_id == "button-previous-subtomogram":
            if current_index == 0:
                new_val = subtomo_keys[-1]
            else:
                new_val = subtomo_keys[subtomo_keys.index(current_val) - 1]
        return subtomo_keys, new_val

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
        State("radio-cleantype", "value"),
        long_callback=True
        # State("dropdown-filetype", "value")
    )
    def read_subtomograms(clicks, filename, contents, num_images, cleaning_type):
        if ctx.triggered_id != "button-read":
            filename = None

        if not filename:
            return "", True, True, False, False, False

        global subtomograms
        subtomograms = dict()

        if cleaning_type == "Clean based on orientation":
            clean_open = [True, False]
        else:
            clean_open = [False, True]

        # ensure temp directory clear
        files = glob.glob(TEMP_FILE_DIR + "*")
        print(files)
        for f in files:
            os.remove(f)

        # data = contents.encode("utf8").split(b";base64,")[1]
        # with open(os.path.join(TEMP_FILE_DIR, filename), "wb") as fp:
        #     fp.write(base64.decodebytes(data))

        save_dash_upload(filename, contents)

        # contents_decoded = io.BytesIO(base64.b64decode(contents))
        temp_file_path = TEMP_FILE_DIR + filename

        if ".mat" in filename:
            try:
                geom = scipy.io.loadmat(temp_file_path, simplify_cells=True)[
                    "subTomoMeta"
                ]["cycle000"]["geometry"]
            except:
                return "Matlab File Unreadable", True, True, False, False, False
        elif ".mod" in filename:
            try:
                imod_data = read_imod(temp_file_path)
            except:
                return "Invalid file", True, True, False, False, False
            subtomo = SubTomogram.tomo_from_imod(Path(filename).stem, imod_data)
            subtomograms[subtomo.name] = subtomo
            return "Tomogram read", False, False, True, *clean_open
        else:
            return "Unrecognised file extension", True, True, False, False, False

        print(geom.keys())

        if num_images == 0:
            counter = 1
        elif num_images == 1:
            counter = 5
        elif num_images == 2:
            counter = -1

        for gkey in geom.keys():
            if counter == 0:
                break
            counter -= 1

            print(gkey)

            subtomo = SubTomogram.tomo_from_mat(gkey, geom[gkey])

            subtomograms[gkey] = subtomo

        return "Tomograms read", False, False, True, *clean_open

    def clean_subtomo(subtomo, clean_params):
        t0 = time()

        subtomo.set_clean_params(clean_params)

        subtomo.reset_cleaning()

        subtomo.autoclean()

        print(subtomo.particle_fate_table())

        print("time for {}:".format(subtomo.name), time() - t0)

        subtomo.generate_particle_df()

    def prox_clean_subtomo(subtomo, dist_min, dist_max):
        if not any(subtomo.reference_points):
            print("Subtomo {} has no reference points uploaded".format(subtomo.name))
        t0 = time()

        subtomo.proximity_clean((dist_min**2, dist_max**2))

        print("time for {}:".format(subtomo.name), time() - t0)

    def save_dash_upload(filename, contents):
        print("Uploading file:", filename)
        data = contents.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(TEMP_FILE_DIR, filename), "wb") as fp:
            fp.write(base64.decodebytes(data))

    @app.callback(
        Output("div-need-refs", "children"),
        Input("collapse-clean", "is_open"),
        Input("upload-ref", "children"),
    )
    def print_needed_refs(_, __):
        global subtomograms
        return [
            tomoname
            for tomoname, subtomo in subtomograms.items()
            if not subtomo.reference_points
        ]

    @app.callback(
        Output("upload-ref", "children"),
        [Input("upload-ref", "filename"), Input("upload-ref", "contents")],
    )
    def upload_refs(filenames, contents):
        global subtomograms

        if not any([filenames]):
            raise PreventUpdate
        for filename, data in zip(filenames, contents):
            if not filename:
                raise PreventUpdate
            save_dash_upload(filename, data)
            try:
                ref_data = read_imod(TEMP_FILE_DIR + filename)
            except:
                return "Unable to read " + filename

            tomo_name = filename.split("-ref")[0]
            print("uploaded: ", tomo_name)
            print("subtomos: ", subtomograms.keys())
            if tomo_name in subtomograms.keys():
                subtomograms[tomo_name].assign_ref_imod(ref_data)
            else:
                return "No matching subtomogram found"

        return filenames

    @app.callback(
        Output("button-next-subtomogram", "disabled"),
        State("inp-dist-min", "value"),
        State("inp-dist-max", "value"),
        State("button-next-subtomogram", "disabled"),
        Input("button-full-prox", "n_clicks"),
        Input("button-preview-prox", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_prox_cleaning(
        dist_min: float, dist_max: float, is_disabled: bool, clicks, clicks2
    ):
        if not all([dist_max, clicks or clicks2]):
            return False
        if not dist_min:
            dist_min = 0.0

        global subtomograms

        for subtomo in subtomograms.values():
            prox_clean_subtomo(subtomo, dist_min, dist_max)

        return not is_disabled

    @app.callback(
        Output("button-next-subtomogram", "disabled"),
        Output("collapse-clean", "is_open"),
        Output("collapse-save", "is_open"),
        State("inp-dist-goal", "value"),
        State("inp-dist-tol", "value"),
        State("inp-ori-goal", "value"),
        State("inp-ori-tol", "value"),
        State("inp-pos-goal", "value"),
        State("inp-pos-tol", "value"),
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
        # subtomo = SubTomogram(name, particles)

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

        global subtomograms

        if ctx.triggered_id == "button-preview-clean":
            print("Preview")
            clean_subtomo(list(subtomograms.values())[0], clean_params)
            return False, True, True
        else:
            print("Full")
            clean_count = 0
            total_subtomos = len(subtomograms.keys())
            for subtomo in subtomograms.values():
                clean_subtomo(subtomo, clean_params)
                clean_count += 1
                print(prog_bar(clean_count, total_subtomos))
        return False, False, True

    size = "50px"

    def inp_num(id_suffix, default=""):
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
                    html.Td("Distance"),
                    inp_num("dist-goal", 25),
                    html.Td("±"),
                    inp_num("dist-tol", 10),
                ]
            ),
            html.Tr(
                [
                    html.Td("Orientation"),
                    inp_num("ori-goal", 9),
                    html.Td("±"),
                    inp_num("ori-tol", 10),
                ]
            ),
            html.Tr(
                [
                    html.Td("Displacement"),
                    inp_num("pos-goal", 90),
                    html.Td("±"),
                    inp_num("pos-tol", 20),
                ]
            ),
            html.Tr([html.Td("Min Neighbours"), inp_num("min-neighbours", 2)]),
            html.Tr([html.Td("CC Threshold"), inp_num("cc-thresh", 5)]),
            html.Tr([html.Td("Min Array Size"), inp_num("array-size", 5)]),
            html.Tr(
                [
                    html.Td("Allow Flipped Particles"),
                    daq.BooleanSwitch(id="switch-allow-flips", on=False),
                ]
            ),
            html.Tr(
                html.Td(
                    dbc.Button(
                        "Preview Cleaning", id="button-preview-clean", color="secondary"
                    ),
                    colSpan=4,
                ),
            ),
            html.Tr(
                html.Td(dbc.Button("Full Cleaning", id="button-full-clean"), colSpan=4),
            ),
        ],
        style={  #'float':'left',
            "overflow": "hidden",
            "margin": "3px",
            "width": "100%"
            #'borderWidth': '1px',
            #'borderStyle': 'dashed',
            #'borderRadius': '5px'
        },
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
                                #'width': '100%',
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
                    html.Td("Files requring references: "),
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
            #'width': '100%',
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

    def collapsing_card(card: dbc.Card, collapse_id: str, start_open: bool = False):
        return dbc.Collapse(
            card,
            id="collapse-" + collapse_id,
            is_open=start_open,
            style={"width": "450px", "height": "100%"},
        )  # ,style={"float":"right"})

    upload_table = html.Table(
        [
            html.Tr([html.Td(upload_file)]),
            html.Tr(
                dcc.Dropdown(
                    [".mat", ".star", ".mod"],
                    id="dropdown-filetype",
                    clearable=False,
                )
            ),
            html.Tr(
                dcc.RadioItems(
                    [
                        "Clean based on orientation",
                        #"Clean based on reference particles",
                    ],
                    "Clean based on orientation",
                    id="radio-cleantype",
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
                    html.Td(dbc.Button("Read Subtomograms", id="button-read")),
                    html.Td(id="label-read"),
                ]
            ),
            # html.Tr(html.Td(dbc.Button("Plot Without Cleaning", id="button-plot_all"))),
        ],
        style={  #'float':'left',
            "overflow": "hidden",
            "margin": "3px",
            "width": "100%"
            #'borderWidth': '1px',
            #'borderStyle': 'dashed',
            #'borderRadius': '5px'
        },
    )

    graph_controls_table = html.Table(
        [
            html.Tr(
                [
                    html.Td("Subtomogram"),
                    dcc.Dropdown(
                        [],
                        id="dropdown-tomo",
                        style={"width": "300px"},
                        clearable=False
                        # value=list(protein_list.keys())[0],
                    ),
                ]
            ),
            html.Tr(
                [
                    html.Td("Cone Plot (experimental)", id="label-cone-plot"),
                    daq.BooleanSwitch(id="switch-cone-plot", on=False),
                ]
            ),
            html.Tr(
                [
                    html.Td("Overall Cone Size", id="label-cone-size"),
                    html.Td(dbc.Input(id="inp-cone-size", value=10, type="number", style={"width":"70%"})),
                    html.Td(dbc.Button("Set", id="button-set-cone-size"))
                ]
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
                            "Previous Subtomogram",
                            id="button-previous-subtomogram",
                        ),
                    )
                ]
            ),
        ],
        style={  #'float':'left',
            "overflow": "hidden",
            "margin": "3px",
            "width": "100%",
            "table-layout":"fixed"
            #'borderWidth': '1px',
            #'borderStyle': 'dashed',
            #'borderRadius': '5px'
        },
    )

    save_table = html.Table(
        [
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
                    html.Td("Also save:"),
                    dcc.Checklist(
                        [".em (Place Object)"],
                        [],
                        inline=True,
                        id="checklist-save-additional",
                    ),
                ]
            ),
            html.Tr(html.Td(dbc.Button("Save Particles", id="button-save"))),
            # dcc.Link("Download Output File", id="link-download", href="a"),
            dcc.Download(id="download-file"),
        ],
        style={  #'float':'left',
            "overflow": "hidden",
            "margin": "3px",
            "width": "100%"
            #'borderWidth': '1px',
            #'borderStyle': 'dashed',
            #'borderRadius': '5px'
        },
    )

    # cleaning_type_div = emptydiv = html.Div(
    #     [
    #         # nonexistent outputs for callbacks with no visible effect
    #         html.Div(id="div-null", style={"display": "none"}),
    #     ],
    #     style={"display": "none"},
    # )

    cleaning_params_card = collapsing_card(
        card("Orientation Cleaning", param_table), "clean"
    )
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
            # dbc.Row([dbc.Col(upload_card, width=4), dbc.Col(cleaning_params_card, width=4), dbc.Col(graph_controls_card, width=4)]),
            dbc.Row(
                html.Td(
                    dbc.Button(
                        "Next Subtomogram",
                        id="button-next-subtomogram",
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
        global subtomograms
        if not t_id:
            return {"display": "none"}
        elif not t_id in subtomograms.keys():
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
        global subtomograms
        if not output_name:
            return None
        if output_name == input_name:
            print(
                "Output and input file cannot be identical, this will lead to loss of data"
            )
            return None

        if ".em (Place Object)" in save_additional:
            write_emfile(subtomograms, "out", keep_selected)
            zip_files(output_name, "em")
            dcc.send_file(TEMP_FILE_DIR + output_name)

        modify_emc_mat(
            subtomograms,
            TEMP_FILE_DIR + output_name,
            TEMP_FILE_DIR + input_name,
            keep_selected,
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
