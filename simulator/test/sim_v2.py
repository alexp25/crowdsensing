# https://www.programmersought.com/article/6871764281/
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
from ortools.constraint_solver import pywrapcp


class CreateDistanceEvaluator():
    def __init__(self, locationData, multiplier=1.3):
        self.distances = {}
        self._multiplier = multiplier 
        
        for from_node in np.arange(len(locationData)):
            self.distances[from_node] = {}
            for to_node in np.arange(len(locationData)):
                if np.alltrue(np.equal(from_node, to_node)):
                    self.distances[from_node][to_node] = 0
                else:
                    self.distances[from_node][to_node] = int(
                            self.getDistanceInKm(locationData[from_node],
                            locationData[to_node]) * self._multiplier)
    
    @staticmethod
    def getDistanceInKm(coord1,coord2):
        # https://en.wikipedia.org/wiki/Haversine_formula
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        if np.isnan(lat1 * lon1 * lat2 * lon2):
            return 0
        
        def deg2rad(deg):
            return deg * (math.pi / 180)
        
        R = 6371  # Earth radius (km)
        dLat = deg2rad(lat2-lat1)  
        dLon = deg2rad(lon2-lon1); 
        a = (   math.sin(dLat/2) * math.sin(dLat/2) 
                + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * 
                  math.sin(dLon/2) * math.sin(dLon/2)    )
    
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) 
        d = R * c 
        return d
    
    def distance_evaluator(self, from_node, to_node):
        """
                 Callback function. In the process of calculating the cost, the pre-calculated distance value is directly extracted to increase the speed.
        """
        return self.distances[from_node][to_node]


class CreateDemandEvaluator():
    def __init__(self, demandData):
        self._demands = demandData
        
    def demand_evaluator(self, from_node, to_node):
        """
                 Callback function
        """
        del to_node
        return self._demands[from_node]


class CreateAllTransitEvaluators():
    def __init__(self, vehicles, distances, serviceTimes):
        """
                 Callback function list
        """
        self._vehicles = vehicles
        self._distances = distances
        self._serviceTimes = serviceTimes
        self.evaluators = []
        # Each car generates a corresponding evaluator according to the speed:
        for v in vehicles.speeds:
            evaluator = CreateOneTransitEvaluator(v, self._distances, 
                                self._serviceTimes).one_transit_evaluator
            self.evaluators.append(evaluator)


class CreateOneTransitEvaluator():
    def __init__(self, speed, distances, serviceTimes):
        self._speed = speed
        self._distances = distances
        self._serviceTimes = serviceTimes
        
    def one_transit_evaluator(self, from_node, to_node):
        """
        Single callback function:
                 Calculate the total amount of time for a single node = distance from the current node to the next node / vehicle speed + service duration of the current node
        """
        if from_node == to_node:
            return 0
        if self._speed == 0:
            return sys.maxsize
        return int(self._distances[from_node][to_node] / self._speed 
                + self._serviceTimes[from_node])


def add_capacity_constraints(routing, listOfConstraints, evaluator, varName):
    
    name = varName
    routing.AddDimensionWithVehicleCapacity(
           evaluator, 
           0, 
           listOfConstraints, 
           True, 
           name)


def add_transit_and_capacity_constraints(routing, listOfConstraints, 
                                         listOfEvaluators, intSlack, varName):
    name = varName
    routing.AddDimensionWithVehicleTransitAndCapacity(
        listOfEvaluators,
        intSlack,
        listOfConstraints,
        False,
        name)


def add_timewindow_constraints(routing, data, varName='time'):
    
    time_dimension = routing.GetDimensionOrDie(varName)
    for node_idx, time_window in enumerate(data.timeWindows):
        if node_idx <= np.max(data.depotIndex):
            continue
        index = routing.NodeToIndex(node_idx)
        servTime = data.serviceTimes[node_idx]
        time_dimension.CumulVar(index).SetRange(
                                    time_window[0], time_window[1]-servTime)
        routing.AddToAssignment(time_dimension.SlackVar(index))
    for veh_idx in np.arange(data.nrVehicles):
        index = routing.Start(veh_idx)
        servTime = data.serviceTimes[data.depotIndex[veh_idx]]
        time_dimension.CumulVar(index).SetRange(
                                        data.earliestWorkHours[veh_idx],
                                        data.latestWorkHours[veh_idx]-servTime)
        routing.AddToAssignment(time_dimension.SlackVar(index))
    for veh_idx in np.arange(len(data.depotIndex)):
        index = routing.End(veh_idx)
        servTime = data.serviceTimes[data.depotIndex[veh_idx]]
        time_dimension.CumulVar(index).SetRange(
                                        data.earliestWorkHours[veh_idx],
                                        data.latestWorkHours[veh_idx]-servTime)



class ConsolePrinter():
    def __init__(self, data, routing, assignment, distances):
        self._data = data
        self._routing = routing
        self._assignment = assignment
        self._distances = distances
        
    def printAll(self):
        total_dist = 0
        total_siteCount = 0
        total_fulfilledDemand = 0
        capacity_dimension = self._routing.GetDimensionOrDie('capacity')
        distance_dimension = self._routing.GetDimensionOrDie('dailyDistance')
        time_dimension = self._routing.GetDimensionOrDie('time')
        siteCount_dimension = self._routing.GetDimensionOrDie('dailyNrJobs')
        
        for vehicle_id in np.arange(self._data.nrVehicles):
            index = self._routing.Start(vehicle_id)
            plan_output = 'Route for person {0}: \n'.format(vehicle_id)
            route_startTime = self._assignment.Value(time_dimension.CumulVar(index))
            route_serviceTime = 0
            route_timeWindow = []
            while not self._routing.IsEnd(index):
                node_index = self._routing.IndexToNode(index)
                next_node_index = self._routing.IndexToNode(
                        self._assignment.Value(self._routing.NextVar(index)))
                step_dist = self._distances[node_index][next_node_index]
                step_load = self._data.demands[node_index]
                step_serviceTime = self._data.serviceTimes[node_index]
                route_serviceTime += step_serviceTime
                step_timewindow = self._data.timeWindows[node_index]
                route_timeWindow.append(step_timewindow)
                time_var = time_dimension.CumulVar(index)
                time_min = self._assignment.Min(time_var)
                time_max = self._assignment.Max(time_var)
                slack_var = time_dimension.SlackVar(index)
                slack_min = self._assignment.Min(slack_var)
                slack_max = self._assignment.Max(slack_var)
                
                plan_output += (
                    ' {node} capacity({capa}) distance({dist}) serviceTime({minTime},{maxTime}) slack({minSlack},{maxSlack})->\n'
                    .format(node=node_index, capa=step_load, dist=step_dist, 
                        minTime=time_min, maxTime=time_max, minSlack=slack_min, 
                        maxSlack=slack_max) )
                index = self._assignment.Value(self._routing.NextVar(index))
            
            end_idx = self._routing.End(vehicle_id) 
            route_endTime = self._assignment.Value(time_dimension.CumulVar(end_idx)) 
            route_dist = self._assignment.Value(distance_dimension.CumulVar(end_idx))
            route_load = self._assignment.Value(capacity_dimension.CumulVar(end_idx))
            route_siteCount = self._assignment.Value(siteCount_dimension.CumulVar(end_idx))
            node_index = self._routing.IndexToNode(index)
            total_dist += route_dist
            total_siteCount += route_siteCount
            total_fulfilledDemand += route_load
            
            plan_output += ' {0} \n'.format(node_index)
            plan_output += ('Objective: minimize vehicle cost + distance cost, maximize number of sites visited\nConstraint:\n 1.vehicle capacity {load} pieces\n 2.vehicle daily distance {dailyDistance} km\n 3.vehicle daily sites {dailySites}\n 4.depot opening hours {depotTime} min\n 5.vehicle shift times {vehicleTime} min\n 6.location time windows {tw}\n'
                    .format(load=self._data.vehicles.capacity[vehicle_id],
                            depotTime=self._data.timeWindows[self._data.depotIndex[vehicle_id]],
                            tw=route_timeWindow,
                            dailyDistance = self._data.vehicles.dailyDistanceLimit[vehicle_id],
                            dailySites = self._data.vehicles.nrJobLimit[vehicle_id],
                            vehicleTime = self._data.vehicles.vehicleTimeWindows[vehicle_id]
                            ) )
            plan_output += 'Result:\n 1.load of the route: {0} pcs\n'.format(route_load)
            plan_output += ' 2.distance of the route: {0} km\n'.format(route_dist)
            plan_output += ' 3.visited nr. sits: {0}\n'.format(route_siteCount)
            plan_output += ' 4.timespan of the route: ({0},{1}) min\n'.format(
                                                                                    route_startTime, route_endTime)
            plan_output += '   of which service time: {0} min\n'.format(route_serviceTime)
            print(plan_output)

        print('Total distance of all routes: {0} km\nTotal nr. visited sites: {1}\nTotal fulfilled demand: {2}\n'
              .format(total_dist,total_siteCount,total_fulfilledDemand))
        print('Dropped nodes: {0}\n').format(self.getDropped())
    
    def getDropped(self):
        dropped = []
        for idx in np.arange(self._routing.Size()):
            if self._assignment.Value(self._routing.NextVar(idx)) == idx:
                dropped.append(idx)
        return dropped



def discrete_cmap(N, base_cmap=None):
    """
    based on https://github.com/google/or-tools/blob/master/examples/python/cvrptw_plot.py
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def build_vehicle_route(routing, assignment, locations, vehicle_id):
    """
    based on https://github.com/google/or-tools/blob/master/examples/python/cvrptw_plot.py
    Build a route for a vehicle by starting at the strat node and
    continuing to the end node.
    """
    
    veh_used = routing.IsVehicleUsed(assignment, vehicle_id)
    if veh_used:
        route = []
        node = routing.Start(vehicle_id)  # Get the starting node index
        route.append(locations[routing.IndexToNode(node)])
        while not routing.IsEnd(node):
          route.append(locations[routing.IndexToNode(node)])
          node = assignment.Value(routing.NextVar(node))
        
        route.append(locations[routing.IndexToNode(node)])
        return route
    else:
        return None


def plot_vehicle_routes(vehicle_routes, ax, data):
    """
    based on https://github.com/google/or-tools/blob/master/examples/python/cvrptw_plot.py
    Plot the vehicle routes on matplotlib axis ax1.
    """
    veh_used = [v for v in vehicle_routes if vehicle_routes[v] is not None]
    cmap = discrete_cmap(len(data.vehicles.names) + 2, 'nipy_spectral')

    for veh_number in veh_used:
        lats, lons = zip(*[(c[0], c[1]) for c in vehicle_routes[veh_number]])
        lats = np.array(lats)
        lons = np.array(lons)

        ax.plot(lons, lats, 'o', mfc=cmap(veh_number + 1))
        ax.quiver(
            lons[:-1],
            lats[:-1],
            lons[1:] - lons[:-1],
            lats[1:] - lats[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=cmap(veh_number + 1))



# Use the default modeling parameters:
model_parameters = pywrapcp.RoutingModel.DefaultModelParameters()
routing = pywrapcp.RoutingModel(
        data.nrLocations, 
        data.nrVehicles, 
        data.depotIndex, 
        data.depotIndex,
        model_parameters)

# Add vehicle cost:
for n,v in enumerate(data.vehicles.costs):
    routing.SetFixedCostOfVehicle(v, n)

# Add distance cost:
distEval = CreateDistanceEvaluator(data.locations)
distance_evaluator = distEval.distance_evaluator # callback function
routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator)

# Add the maximum distance travel constraint for each vehicle:
add_capacity_constraints(routing, data.vehicles.dailyDistanceLimit,
                         distance_evaluator, 'dailyDistance')    

# Add the maximum capacity constraint per vehicle:
demand_evaluator = CreateDemandEvaluator(data.demands).demand_evaluator
add_capacity_constraints(routing, data.vehicles.capacity, 
                         demand_evaluator, 'capacity')

# Add the maximum number of access points per vehicle:
nrJobs_evaluator = CreateDemandEvaluator(
                                data.visitedLocations).demand_evaluator
add_capacity_constraints(routing, data.vehicles.nrJobLimit,
                         nrJobs_evaluator, 'dailyNrJobs')

# Add runtime constraints
transitEval = CreateAllTransitEvaluators(data.vehicles, distEval.distances, 
                                         data.serviceTimes)
add_transit_and_capacity_constraints(routing, data.latestWorkHours, 
                                 transitEval.evaluators, 
                                 int(np.max(data.latestWorkHours)), 'time')

# Add time window constraint
add_timewindow_constraints(routing, data, 'time')

# Set the search strategy (in this case, the default parameters are used mainly, and other parameters refer to the google tutorial):
# https://developers.google.com/optimization/routing/routing_options
search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# Set a penalty to allow customer sites not to be accessed (when all vehicles reach the upper limit of the constraint)
# Otherwise the problem may be unsolvable.
non_depot = set(range(data.nrLocations))
non_depot.difference_update(data.depotIndex)
penalty = 400000
nodes = [routing.AddDisjunction([c], penalty) for c in non_depot]

# 
assignment = routing.SolveWithParameters(search_parameters)

#  
printer = ConsolePrinter(data, routing, assignment, distEval.distances)
printer.printAll()

# plot routes
dropped = []
dropped = printer.getDropped()

vehicle_routes = {}
for veh in np.arange(data.nrVehicles):
    vehicle_routes[veh] = build_vehicle_route(routing, assignment, 
                                                      data.locations, veh)
# Plotting of the routes in matplotlib.
fig = plt.figure()
ax = fig.add_subplot(111)
# Plot all the nodes as black dots.
clon, clat = zip(*[(c[0], c[1]) for i,c in enumerate(data.locations) if i not in dropped])
ax.plot(clat, clon, 'k.')
# plot the routes as arrows
plot_vehicle_routes(vehicle_routes, ax, data)

