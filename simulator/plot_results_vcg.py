"""Capacited Vehicles Routing Problem (CVRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import yaml
import csv
import traceback
import json
import math

import numpy as np

from modules import graph
from modules import plot_routes
import compute_geometry
from modules import plotter


def create_data_model():
    global config, input_data
    """Stores the data for the problem."""
    data = {}

    input_file = None
    with open('config.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        input_file = config["input_file"]

    depots = []
    issues = []
    vehicles = []
    with open(input_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        input_data = yaml.load(file, Loader=yaml.FullLoader)
        print(config)
        depots = input_data["depots"]
        issues = input_data["issues"]
        vehicles = input_data["vehicles"]

    # config["matrix_file"] = "dm_test.txt"

    with open(config["matrix_file"]) as file:
        csvdata = csv.reader(file)
        dm = []
        nrows = 0

        for row in csvdata:
            r = [int(e) for e in row]
            ncols = len(r)
            dm.append(r)
            nrows += 1

        print("msize: ", nrows, ncols)

    data['distance_matrix'] = dm

    data['demands'] = [int(d) * input_data["options"]["demand_factor"]
                       for d in input_data["demands"]]

    data['num_vehicles'] = len(vehicles)
    data['vehicle_capacities'] = [int(v["capacity"]) for v in vehicles]
    data['vehicle_fuel'] = [int(v["fuel"]) for v in vehicles]

    sum_demands = sum(data['demands'])
    sum_capacities = sum(data['vehicle_capacities'])

    # ratio = sum_demands / sum_capacities
    # if ratio > 1:
    #     data["vehicle_capacities"] = [int(d*ratio) for d in data["vehicle_capacities"]]

    data['depot'] = 0

    # So far, we have assumed that all vehicles start and end at a single location, the depot.
    # You can also set possibly different start and end locations for each vehicle in the problem.
    # To do so, pass two vectors, containing the indices of the start and end locations,
    # as inputs to the RoutingModel method in the main function.
    # Here's how to create the start and end vectors in the data section of the program:

    data['starts'] = [i for i in range(len(depots))]
    data['ends'] = [i for i in range(len(depots))]

    print(data)

    print("sum demands: " + str(sum_demands))
    print("sum capacities: " + str(sum_capacities))

    # quit()
    return data


def get_average_specs(route_vect):
    """get average specs for simulation scenario e.g. demand_factor = 5"""
    full_specs = json.loads(json.dumps(route_vect[0]))
    avg_specs = json.loads(json.dumps(route_vect[0]))
    var_specs = json.loads(json.dumps(route_vect[0]))

    # init full specs
    for spec in full_specs:
        for k in spec:
            spec[k] = []

    for route in route_vect[1:]:
        for i, spec in enumerate(avg_specs):
            for k in spec:
                # add specs
                avg_specs[i][k] += route[i][k]
                # append specs
                full_specs[i][k].append(route[i][k])

    # for each vehicle
    for i, spec in enumerate(avg_specs):
        # for each spec (e.g. load, points, distance)
        for k in spec:
            # compute average
            avg_specs[i][k] /= len(route_vect)
            # compute variance
            # https://stackabuse.com/calculating-variance-and-standard-deviation-in-python/
            var_specs[i][k] = math.sqrt(
                sum([(x - avg_specs[i][k]) ** 2 for x in full_specs[i][k]]) / len(route_vect))

    return avg_specs, var_specs, full_specs


def main():
    """plot results."""

    # set config
    plot_avg = True
    # plot_avg = False

    plot_type = "load"
    plot_name = "load"
    plot_type = "points"
    plot_name = "points"
    # plot_type = "dist"
    # plot_name = "dist"
    

    raw_mode = True

    alpha = None
    alpha = 1
    alpha = 0.9

    # Instantiate the data problem.
    data = create_data_model()

    routes_vect = []

    mat_chart = []

    with open("data/routes_sim_vcg_multi.adapted.txt", "r") as f:
        disp_routes = f.read()
        disp_routes = json.loads(disp_routes)
        routes_vect = disp_routes

    for route in routes_vect:
        vehicles = []
        avg_loads = []
        avg_dist = []
        avg_points = []
        stdev_loads = []
        stdev_dist = []
        stdev_points = []

        for v in input_data["vehicles"]:
            found = False
            found_bid = None
            for bid in route:
                if bid["vehicle"] == v["id"]:
                    found = True
                    found_bid = bid
                    break
                if not found:
                    found_bid = {
                        "load": 0,
                        "points": 0,
                        "distance": 0
                    }

            avg_loads.append(found_bid["load"] * 1.0)
            # avg_points.append(found_bid["points"] * 100)
            avg_points.append(found_bid["points"] * 1.0)
            avg_dist.append(found_bid["distance"])

        # print(avg_loads)
        if plot_type == "load":
            mat_chart.append(avg_loads if plot_avg else stdev_loads)
        elif plot_type == "points":
            mat_chart.append(avg_points if plot_avg else stdev_points)
        elif plot_type == "dist":
            mat_chart.append(avg_dist if plot_avg else stdev_dist)

    # print(mat_chart)
    # quit()
    mat_chart_np = np.array(mat_chart)

    mat_chart_np = np.transpose(mat_chart_np)

    shape = np.shape(mat_chart_np)

    print(mat_chart_np)

    if alpha is not None:
        if alpha == 1:
            mat = mat_chart_np
            # average
            for i in range(shape[0]):
                avg_row = mat[i,:].mean()
                print(avg_row)
                for j in range(shape[1]):
                    mat[i][j] = avg_row
            mat_chart_np = mat
            print(mat_chart_np)
        else:
            # filter
            mat = mat_chart_np
            print(shape)
            for i in range(shape[0]):
                for j in range(1, shape[1]):
                    mat[i][j] = mat[i][j-1] * alpha + mat[i][j] * (1-alpha)
            mat_chart_np = mat
            print(mat_chart_np)

    mat_chart = mat_chart_np.tolist()
    # quit()

    labels = ["p" + str(i+1) + " (" + str(d) + ")" for i,
              d in enumerate(data['vehicle_capacities'])]
    colors = [None for d in data['vehicle_capacities']]

    colors = plotter.create_discrete_cmap(data["vehicle_capacities"])
    colors = [colors(i+1) for i in range(len(data['vehicle_capacities']))]
    colors = list(reversed(colors))

    fig = graph.plot_timeseries_multi(mat_chart, range(shape[1]), labels, colors,
                                      "VCG load balancing", "simulation no.", "average " + plot_name if plot_avg else "stdev " + plot_name, False)
    fig.savefig("figs/results_vcg_"+plot_type + "_" + ("avg" if plot_avg else "stdev") +
                ("_raw" if raw_mode else "") + ".png", dpi=300)


if __name__ == '__main__':
    main()
