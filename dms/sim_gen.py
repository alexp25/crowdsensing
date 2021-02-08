"""Capacited Vehicles Routing Problem (CVRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import yaml
import csv
import traceback
import json

from modules import plot_routes, geometry
import compute_geometry
import config_loader

from modules import shuffle

n_iter = 100


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
                "demand": demands[i]
            }

        find_index = 0

        for i in range(n_iter):
            if i == 0:
                distance_matrix = compute_geometry.compute_distance_matrix_wrapper()
            else:
                distance_matrix = compute_geometry.get_distance_matrix_with_random_depots()

            print("iteration: " + str(i) + " with demand factor: " + str(df) + " [" +
                  str(int((i_df * n_iter + i)/(len(demand_factor_range) * n_iter)*100)) + "%]")

            # each agent covers a given range (treasure scan)
            for i, v in enumerate(vehicles):
                print("vehicle: ", i)
                # check nearby places within range
                found_places = []
                for j, d in enumerate(distance_matrix[i]):
                    if j >= n_vehicles:
                        # print(j,d)
                        if d <= options["scan_radius"]:
                            print("found: ", d)
                            issue = issues[j-n_vehicles]
                            dict_issue = fill_dict[issue]

                            # check if place was already found by another agent
                            if not dict_issue["found"]:
                                dict_issue["found"] = True
                                dict_issue["finder"] = vehicles[i]["id"]
                                dict_issue["find_index"] = find_index

                                found_places.append(dict_issue)

                                find_index += 1

                # assign random items for each agent and found places
                n_slots = 0
                for fps in found_places:
                    # check filled items, get number of free slots
                    if not fps["filled"]:
                        n_slots += fps["demand"]

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

                print("n_items: ", n_items)
                print("filled: ", slots_index)

                # check unfilled slots, fill with last item type
                n_unfilled = n_slots - slots_index
                if slots_index < n_slots:
                    for n in range(n_unfilled):
                        slots[slots_index + n] = items[len(items)-1]["item"]

                # shuffle items/slots
                slots = shuffle.fisher_yates_shuffle_improved(slots)
                print(len(slots))

                slots_index = 0

                # assign items to actual places
                for fps in found_places:
                    for d in range(fps["demand"]):
                        if not fps["filled"]:
                            fps["items"].append(slots[slots_index])
                            slots_index += 1
                        else:
                            print("already filled")

        # count end result, number of items by type
        items_result_dict = {}
        total_filled = 0
        total_found = 0

        for item in items:
            items_result_dict[item["item"]] = 0

        # check filled places
        for fd in fill_dict:
            issue = fill_dict[fd]
            if issue["found"]:
                total_found += 1
            # check filled items for each place
            for item in issue["items"]:
                if item in items_result_dict:
                    items_result_dict[item] += 1
                    total_filled += 1
                else:
                    items_result_dict[item] = 1
                    total_filled += 1

        # print(fill_dict[fd])
        print(items_result_dict)
        print("places found: ", total_found, "/", len(issues))
        print("filled demand: ", total_filled, "/", sum(demands))

        items_ratio_dict = {}
        for item in items_result_dict:
            items_ratio_dict[item] = items_result_dict[item] / total_filled

        print(items_ratio_dict)

        # with open("data/routes_sim." + str(df) + ".txt", "w") as f:
        #     f.write(disp_routes)

        # with open("data/routes_sim_info." + str(df) + ".txt", "w") as f:
        #     sim_info = {
        #         "demand_factor": df
        #     }
        #     f.write(json.dumps(sim_info))


if __name__ == '__main__':
    main()
