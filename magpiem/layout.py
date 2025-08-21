# -*- coding: utf-8 -*-
"""
Layout components for the MagpiEM Dash application.
"""

import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc, html

from .classes import simple_figure

EMPTY_FIG = simple_figure()


def inp_num(id_suffix, size="50px"):
    """Create a numeric input table cell."""
    return html.Td(
        dbc.Input(
            id="inp-" + id_suffix,
            type="number",
            style={"appearance": "textfield", "width": size},
        ),
        style={"width": size},
    )


def create_param_table():
    """Create the cleaning parameters table."""
    return html.Table(
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


def create_upload_file():
    """Create the file upload component."""
    return dcc.Upload(
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


def create_upload_table():
    """Create the upload table with file controls."""
    return html.Table(
        [
            html.Tr([html.Td(create_upload_file())]),
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


def create_graph_controls_table():
    """Create the graph controls table."""
    return html.Table(
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
                            value=15,
                            type="number",
                            style={"width": "70%"},
                        )
                    ),
                    html.Td(dbc.Button("Set", id="button-set-cone-size")),
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


def create_save_table():
    """Create the save results table."""
    return html.Table(
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


def create_card(title: str, contents):
    """Create a Bootstrap card component."""
    return dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody([contents])],
        style={
            "width": "100%",
            "margin": "5px",
            "height": "100%",
        },
    )


def create_collapsing_card(
    display_card: dbc.Card, collapse_id: str, start_open: bool = False
):
    """Create a collapsible card component."""
    return dbc.Collapse(
        display_card,
        id="collapse-" + collapse_id,
        is_open=start_open,
        style={"width": "450px", "height": "100%"},
    )


def create_graph():
    """Create the main graph component."""
    return dcc.Graph(
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


def create_empty_div():
    """Create the empty div for hidden outputs."""
    return html.Div(
        [
            html.Div(id="div-null", style={"display": "none"}),
        ],
        style={"display": "none"},
    )


def create_store_components():
    """Create all the dcc.Store components."""
    return html.Div(
        [
            dcc.Store(id="store-lattice-data"),
            dcc.Store(id="store-tomogram-data"),
            dcc.Store(id="store-selected-lattices"),
            dcc.Store(id="store-clicked-point"),
            dcc.Store(id="store-camera"),
            dcc.Store(id="store-last-click", data=0.0),
            dcc.Store(id="store-session-key", data=""),
            dcc.Store(id="store-cache-cleared", data=False),
        ]
    )


def create_main_layout():
    """Create the main application layout."""
    # Create cards
    cleaning_params_card = create_collapsing_card(
        create_card("Cleaning", create_param_table()),
        "clean",
    )
    upload_card = create_collapsing_card(
        create_card("Choose File", create_upload_table()), "upload", start_open=True
    )
    graph_controls_card = create_collapsing_card(
        create_card("Graph Controls", create_graph_controls_table()), "graph-control"
    )
    save_card = create_collapsing_card(
        create_card("Save Result", create_save_table()), "save"
    )

    return html.Div(
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
            dbc.Row(
                html.Div(
                    id="div-graph-data",
                    style={
                        "minHeight": "80px",
                        "height": "80px",
                        "overflow": "auto",
                        "padding": "12px",
                        "margin": "8px 0",
                    },
                )
            ),
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
                    dcc.Interval(
                        id="interval-clear-points", interval=200, disabled=True
                    ),
                ]
            ),
            dbc.Row([create_graph()]),
            create_store_components(),
            html.Footer(
                html.Div(
                    dcc.Link(
                        "Documentation and instructions",
                        href="https://github.com/fnight128/MagpiEM",
                        target="_blank",
                    )
                )
            ),
            create_empty_div(),
            dcc.ConfirmDialog(
                id="confirm-cant-save-progress",
                message="Progress can only be saved if cleaning was run on all tomograms.",
            ),
        ],
    )
