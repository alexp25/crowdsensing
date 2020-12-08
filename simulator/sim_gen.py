"""Capacited Vehicles Routing Problem (CVRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import yaml
import csv
import traceback

from modules import plot_routes


def create_data_model():
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


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    # manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
    #                                        data['num_vehicles'], data['depot'])

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'],
                                           data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Add the maximum distance travel constraint for each vehicle:
    routing.AddDimensionWithVehicleCapacity(
        transit_callback_index,
        0,  # null capacity slack
        data['vehicle_fuel'],  # vehicle fuel / max daily/travel distance
        True,  # start cumul to zero
        'DailyDistance')

    # Allow to drop nodes.
    penalty = 10000
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    time_limit = 1
    search_parameters.time_limit.FromSeconds(time_limit)

    # Total Distance of all routes: 18041m
    # Total Load of all routes: 47

    print("running solver max time limit: " + str(time_limit))
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        routes = plot_routes.print_solution(data, manager, routing, assignment)
        print(routes)


if __name__ == '__main__':
    main()
