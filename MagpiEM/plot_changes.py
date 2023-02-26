# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:39:39 2023

@author: Frank
"""

from classes import SubTomogram, Particle

import colorsys
import scipy.io
import plotly.graph_objects as go
from collections import defaultdict

BLACK = "#000000"
MAT_FILE = "nucleosome_tomo14.mat"
TOMO_NAME = "lamellae_014_1"

#mat_data = scipy.io.loadmat(MAT_FILE, simplify_cells=True)["subTomoMeta"]

try:
    mat_data = scipy.io.loadmat(MAT_FILE, simplify_cells=True)["subTomoMeta"]
except:
    assert False, ".mat file unreadable"
    
print(mat_data["cycle001"].keys())

def colour_range(num_points):
    HSV_tuples = [(x * 1.0 / num_points, 0.75, 0.75) for x in range(num_points)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return [
        "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
        for (r, g, b) in RGB_tuples
    ]

tomo_list = set(mat_data["cycle000"]["geometry"].keys())

assert TOMO_NAME in tomo_list, "Tomo not found in file. Available tomos:" + ",".join(tomo_list)

cycles = [key for key in mat_data if "cycle" in key]
cycles.remove("cycle000")

num_cycles = len(cycles)

col_range = colour_range(num_cycles)


cycle_col_dict = {cycle: col_range[ind] for ind, cycle in enumerate(cycles)}

tomo_cones = []

tomo_cycles_dict = {}



def cone_trace(df, colour, opacity=1, text="", sizeref=10):
    return go.Cone(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        u=df["u"],
        v=df["v"],
        w=df["w"],
        text=text,
        sizemode="scaled",
        sizeref=sizeref,
        colorscale=[[0, colour], [1, colour]],
        showscale=False,
        opacity=opacity,
    )

def scatter3d_trace(df, colour, opacity=1, text=""):
    return go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        text=text,
        marker=dict(size=6, color=colour, opacity=opacity),
        showlegend=False,
    )

for cycle, cycle_data in mat_data.items():
    if not "cycle" in cycle: continue
    print("cycle", cycle)
    initial_picking = cycle == "cycle000"
    geom_key = "geometry" if initial_picking else "Avg_geometry"

    tomo_data = cycle_data[geom_key][TOMO_NAME]

    tomo_read = SubTomogram.tomo_from_mat(cycle, tomo_data)
    
    tomo_read.all_particles = {particle for particle in tomo_read.all_particles if particle.particle_id % 100 == 0}
    
    tomo_df = tomo_read.all_particles_df()
    
    
    col = BLACK if initial_picking else cycle_col_dict[cycle]
    tomo_cone = scatter3d_trace(tomo_df, col, text=cycle)#, sizeref=3)
    tomo_cones.append(tomo_cone)

fig = go.Figure()

for tcone in tomo_cones:
    fig.add_trace(tcone)

fig.write_html("res.html")
