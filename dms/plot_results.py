import yaml
import csv
import traceback
import json
import math

import numpy as np
from modules import graph
from modules import plotter
from modules import loader

results_filename = "dms_results_3.csv"
results_photo = "dms_fill_3"

results = loader.read_csv2(results_filename)
print(results)
series = ['T', 'S', 'M']
xlabels = [str(d+1) for d in results['epoch']]
colors = [None for d in range(len(series))]
print(xlabels)

colors = plotter.create_discrete_cmap(series)
colors = [colors(i+1) for i in range(len(series))]
colors = list(reversed(colors))

print(colors)

mat_chart = [results["T"], results["S"], results["A"]]

# dynamic map search
fig = graph.plot_timeseries_multi(
    mat_chart, xlabels, series, colors, "DMS fill", "epoch", "distribution", False)
fig.savefig("results_" + results_photo + ".png", dpi=300)
# mat_chart = [results["iterations"]]
# fig = graph.plot_timeseries_multi(mat_chart, xlabels, series, colors, "DMS fill", "epoch", "distribution", False)
