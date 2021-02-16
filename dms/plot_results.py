import yaml
import csv
import traceback
import json
import math

import numpy as np
from modules import graph
from modules import plotter
from modules import loader

number = 6

results_filename = "./data/dms_results" + \
    ("_" + str(number) if number is not None else "") + ".csv"
results_photo = "./data/dms_fill" + \
    ("_" + str(number) if number is not None else "")
map_filename = "./data/dms_map" + \
    ("_" + str(number) if number is not None else "") + ".json"
map_photo = "./data/dms_map" + \
    ("_" + str(number) if number is not None else "") + ".png"

results = loader.read_csv2(results_filename)
print(results)
series = ['T', 'C', 'S', 'M']
xlabels = [str(d+1) for d in results['epoch']]
colors = [None for d in range(len(series))]
print(xlabels)

series_ext = series + ["P"]
colors = plotter.create_discrete_cmap(series)
colors_plot = [colors(i+1) for i in range(len(series))]
colors_plot = list(reversed(colors_plot))

colors = plotter.create_discrete_cmap(series_ext)
colors_map = [colors(i+1) for i in range(len(series_ext))]
colors_map = list(reversed(colors_map))
# colors = list(colors)

print(colors)

mat_chart = [results["T"], results["C"], results["S"], results["A"]]

# colors_plot = ['yellow', 'blue', 'orange']

# dynamic map search
fig = graph.plot_timeseries_multi(
    mat_chart, xlabels, series, colors_plot, "DMS fill", "epoch", "distribution", False)
fig.savefig(results_photo, dpi=300)

# mat_chart = [results["iterations"]]
# fig = graph.plot_timeseries_multi(mat_chart, xlabels, series, colors, "DMS fill", "epoch", "distribution", False)

map_geometry = loader.read_json(map_filename)
place_coords = []
item_coords = {
    'T': [],
    'C': [],
    'S': [],
    'A': []
}
for place in map_geometry:
    place_coords.append(place["coords"])
    for item in place["items"]:
        t = item["type"]
        if t in item_coords:
            item_coords[t].append(item["coords"])
        else:
            item_coords[t] = [item["coords"]]
fig = plotter.plot_vehicle_routes_wrapper(
    None, place_coords, item_coords, colors_plot + ['cyan'], None, None, 0.1)
fig.savefig(map_photo, dpi=300)
