"""Capacited Vehicles Routing Problem (CVRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = [[0, 4697, 3212, 1180, 4024, 2873, 3476, 4060, 1266, 3370, 3778, 2807, 2338], [2667, 0, 2685, 2746, 4279, 1470, 2621, 365, 3697, 2844, 1818, 1604, 2795], [2258, 2082, 0, 2336, 3884, 1075, 2226, 2262, 3288, 430, 2028, 1195, 2386], [1668, 3628, 3636, 0, 4292, 2622, 3744, 3808, 1534, 3795, 4202, 2555, 2606], [2358, 2455, 2462, 1657, 0, 1448, 2599, 2635, 3388, 2621, 3028, 1382, 2573], [4630, 2202, 4734, 3929, 2808, 0, 1151, 2481, 3791, 4893, 3978, 3654, 4845], [3832, 1501, 3937, 3132,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            2011, 2923, 0, 1780, 3676, 4095, 3277, 2856, 4047], [2848, 365, 2865, 2926, 4459, 1651, 2801, 0, 3877, 3024, 1998, 1785, 2976], [2296, 3276, 4529, 2105, 2603, 3515, 2055, 3555, 0, 4688, 5096, 3449, 3234], [1827, 1652, 1845, 1906, 3454, 645, 1796, 1832, 2857, 0, 1597, 764, 1956], [2561, 1102, 2337, 2640, 4173, 1364, 2515, 1282, 3591, 2496, 0, 1498, 2689], [2675, 3183, 1515, 3619, 5128, 2319, 3470, 3364, 3705, 1674, 2082, 0, 2556], [2423, 3974, 3084, 3367, 6479, 2968, 4119, 4155, 3453, 3242, 3927, 2901, 0]]
    data['demands'] = [4, 2, 1, 8, 2, 1, 12, 4, 8, 2]

    print(data['demands'])
    data['vehicle_capacities'] = [50, 80, 100]
    data['vehicle_fuel'] = [e*19900 for e in [1, 1, 1]]
    data['num_vehicles'] = 3

    data['depot'] = 0

    # So far, we have assumed that all vehicles start and end at a single location, the depot.
    # You can also set possibly different start and end locations for each vehicle in the problem.
    # To do so, pass two vectors, containing the indices of the start and end locations,
    # as inputs to the RoutingModel method in the main function.
    # Here's how to create the start and end vectors in the data section of the program:

    data['starts'] = [0, 1, 2]
    data['ends'] = [0, 1, 2]

    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
    print(dropped_nodes)
    # Display routes
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle ' + \
            str(vehicle_id) + \
            " [" + str(data['vehicle_capacities'][vehicle_id]) + ']\n'
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total Distance of all routes: {}m'.format(total_distance))
    print('Total Load of all routes: {}'.format(total_load))


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
    penalty = 1000
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)


if __name__ == '__main__':
    main()
