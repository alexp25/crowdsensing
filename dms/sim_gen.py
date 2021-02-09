
import yaml
import csv
import traceback
import json

from modules import geometry
import compute_geometry
import config_loader

from modules import shuffle
import numpy as np
import pandas as pd

n_iter = 100
use_initial_depots = False
disp_view = False
use_external_input = False
use_external_input = True


def main():
    # Instantiate the data problem.
    config, input_data, coords, dm = config_loader.load_config()
    depots = input_data["depots"]
    issues = input_data["issues"]
    vehicles = input_data["vehicles"]
    n_vehicles = len(vehicles)
    options = input_data["options"]
    demands = [int(d) * options["demand_factor"]
               for d in input_data["demands"]]
    vehicle_capacities = [int(v["capacity"]) for v in vehicles]
    vehicle_fuel = [int(v["fuel"]) for v in vehicles]

    items = options["items"]

    print(items)

    if options["sort_order_msb"]:
        items.sort(key=lambda x: x["weight"], reverse=True)

    print(items)

    sum_demands = sum(demands)
    sum_capacities = sum(vehicle_capacities)

    depot = 0

    # So far, we have assumed that all vehicles start and end at a single location, the depot.
    # You can also set possibly different start and end locations for each vehicle in the problem.
    # To do so, pass two vectors, containing the indices of the start and end locations,
    # as inputs to the RoutingModel method in the main function.
    # Here's how to create the start and end vectors in the data section of the program:

    start_points = [i for i in range(len(depots))]
    end_points = [i for i in range(len(depots))]

    print("sum demands: " + str(sum_demands))
    print("sum capacities: " + str(sum_capacities))

    # compute_geometry.load_config()
    n_iter = options["n_iter"]
    n_epochs = options["n_epochs"]

    if use_external_input:
        df = pd.read_csv('coords_nearby_filtered.csv')

        place_ids = []
        coords = []
        place_ids = df["google_id"]
        coords_lat = [lat for lat in df["lat"]]
        coords_lng = [lng for lng in df["lng"]]
        for i in range(len(coords_lat)):
            coords.append([coords_lat[i], coords_lng[i]])
        place_ids = ["place_id:" + pid for pid in place_ids]
        print(place_ids[0])
        print(coords[0])
        issues = place_ids[n_vehicles:]
        input_data["demands"] = [5 for p in place_ids]

        compute_geometry.set_coords(coords)

    # quit()

    if options["use_range"]:
        # use demand factor range
        demand_factor_range = options["demand_factor_range"]
        demand_factor_range = range(
            demand_factor_range[0], demand_factor_range[1] + 1)
        print("using demand factor range: ", list(demand_factor_range))
    else:
        # use specified demand factor
        demand_factor_range = [options["demand_factor"]]
        print("using default demand factor: ", list(demand_factor_range))

    geometry.init_random(False)

    epoch_results_vect = []

    for epoch in range(n_epochs):
        for i_df, df in enumerate(demand_factor_range):
            demands = [int(d) * df
                       for d in input_data["demands"]]
            demands = demands[n_vehicles:]

            fill_dict = {}

            for i, issue in enumerate(issues):
                fill_dict[issue] = {
                    "place": issue,
                    "items": [],
                    "found": False,
                    "filled": False,
                    "finder": None,
                    "find_index": -1,
                    "total_revisits": 0,
                    "demand": demands[i]
                }

            print(len([k for k in fill_dict]))

            find_index = 0

            for i in range(n_iter):
                if i == 0 and use_initial_depots:
                    distance_matrix = compute_geometry.compute_distance_matrix_wrapper()
                else:
                    distance_matrix = compute_geometry.get_distance_matrix_with_random_depots()

                print("epoch: " + str(epoch) + ", iteration: " + str(i) + " with demand factor: " + str(df) + " [" +
                      str(int((i_df * n_iter + i)/(len(demand_factor_range) * n_iter)*100)) + "%]")

                # each agent covers a given range (treasure scan)
                for i_vehicle, v in enumerate(vehicles):
                    if disp_view:
                        print("vehicle: ", i_vehicle)
                    # check nearby places within range
                    found_places = []
                    for j, d in enumerate(distance_matrix[i_vehicle]):
                        if j >= n_vehicles:
                            # print(j,d)
                            if d <= options["scan_radius"]:
                                if disp_view:
                                    print("found: ", d)
                                issue = issues[j-n_vehicles]
                                dict_issue = fill_dict[issue]

                                # check if place was already found by another agent
                                if not dict_issue["found"]:
                                    dict_issue["found"] = True
                                    dict_issue["finder"] = vehicles[i_vehicle]["id"]
                                    dict_issue["find_index"] = find_index
                                    found_places.append(dict_issue)
                                    find_index += 1
                                else:
                                    dict_issue["total_revisits"] += 1
                                    if disp_view:
                                        print("already found: ", d)

                    # assign random items for each agent and found places
                    n_slots = 0
                    for fps in found_places:
                        # check filled items, get number of free slots
                        if not fps["filled"]:
                            n_slots += fps["demand"]

                    if disp_view:
                        print("n_slots: ", n_slots)

                    # generate free slots
                    slots = [None] * n_slots
                    n_items = []

                    # compute number of slots for each item type
                    for item in items:
                        n_items.append(int(n_slots * item["weight"]))

                    # assign items for free slots
                    slots_index = 0
                    for k, n_item in enumerate(n_items):
                        for n in range(n_item):
                            slots[slots_index] = items[k]["item"]
                            slots_index += 1

                    if disp_view:
                        print("n_items: ", n_items)
                        print("filled: ", slots_index)

                    # check unfilled slots, fill with last item type
                    n_unfilled = n_slots - slots_index
                    if slots_index < n_slots:
                        for n in range(n_unfilled):
                            slots[slots_index +
                                  n] = items[len(items)-1]["item"]

                    # shuffle items/slots
                    slots = shuffle.fisher_yates_shuffle_improved(slots)

                    if disp_view:
                        print(len(slots))

                    slots_index = 0

                    # assign items to actual places
                    for fps in found_places:
                        for d in range(fps["demand"]):
                            if not fps["filled"]:
                                fps["items"].append(slots[slots_index])
                                slots_index += 1
                            else:
                                if disp_view:
                                    print("already filled")

                if check_results(items, issues, demands, fill_dict, i, epoch, False)[0]:
                    break

            _, epoch_results = check_results(
                items, issues, demands, fill_dict, i, epoch, True)
            epoch_results_vect.append(epoch_results)

    output_str = "epoch,places,demand,T,S,A,revisits,iterations\n"
    for epoch_results in epoch_results_vect:
        output_str += epoch_results + "\n"
    print(output_str)
    with open("dms_results.csv", "w") as f:
        f.write(output_str)


def check_results(items, issues, demands, fill_dict, iteration, epoch, disp):
    # count end result, number of items by type
    items_result_dict = {}
    total_filled = 0
    total_found = 0
    total_revisits = 0

    output_str = ""

    for item in items:
        items_result_dict[item["item"]] = 0

    # check filled places
    for fd in fill_dict:
        issue = fill_dict[fd]
        if issue["found"]:
            total_found += 1
        total_revisits += issue["total_revisits"]
        # check filled items for each place
        for item in issue["items"]:
            if item in items_result_dict:
                items_result_dict[item] += 1
                total_filled += 1
            else:
                items_result_dict[item] = 1
                total_filled += 1

    if disp:
        print(items_result_dict)
        output_str += str(epoch) + "," + str(total_found) + \
            "," + str(total_filled) + ","
        print("places found: ", total_found, "/", len(issues))
        print("filled demand: ", total_filled, "/", sum(demands))

    items_ratio_dict = {}
    if total_filled != 0:
        for item in items_result_dict:
            items_ratio_dict[item] = items_result_dict[item] / total_filled

    if disp:
        print(items_ratio_dict)
        for k in items_ratio_dict:
            output_str += str(items_ratio_dict[k])+","
        output_str += str(total_revisits) + "," + str(iteration+1)
        print("completed in ", iteration+1, " iterations")

    return total_filled == sum(demands), output_str


if __name__ == '__main__':
    main()
