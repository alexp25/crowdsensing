
import numpy as np
from matplotlib import pyplot as plt
from modules import graph

# https://github.com/google/or-tools/blob/stable/examples/python/cvrptw_plot.py


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def create_discrete_cmap(data):
    cmap = discrete_cmap(len(data) + 2, 'nipy_spectral')
    return cmap

def plot_vehicle_routes(veh_route, ax1, customers, starts, ends, annotate):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    Args: veh_route (dict): a dictionary of routes keyed by vehicle idx.  ax1
    (matplotlib.axes._subplots.AxesSubplot): Matplotlib axes  customers
    (Customers): the customers instance.  vehicles (Vehicles): the vehicles
    instance.
  """

    print(veh_route)
    veh_used = [v for v in veh_route if veh_route[v] is not None]

    # print(starts)

    # print(veh_used)

    # quit()

    cmap = discrete_cmap(len(starts) + 2, 'nipy_spectral')

    nv = len(veh_used)

    for i in range(len(customers) - nv):
        ax1.annotate(
            'T{veh}'.format(veh=i+1),
            # lng, lat
            xy=(customers[i+nv][0], customers[i+nv][1]),
            xytext=(10, 10),
            xycoords='data',
            textcoords='offset points',
            fontsize=15
        )

    for veh_number in veh_used:

        lats, lons = zip(*[(c[1], c[0]) for c in veh_route[veh_number]])
        lats = np.array(lats)
        lons = np.array(lons)
        s_dep = customers[starts[veh_number]]
        s_fin = customers[ends[veh_number]]

        # for
        if annotate:
            ax1.annotate(
                'P{veh}'.format(veh=veh_number+1),
                # lng, lat
                xy=(s_dep[0], s_dep[1]),
                xytext=(10, 10),
                xycoords='data',
                textcoords='offset points',
                fontsize=15
            )

        ax1.plot(lons, lats, 'bs', mfc=cmap(veh_number + 1), markersize=10)
        # 'b*'
        ax1.plot([s_dep[0]], [s_dep[1]], 'bo',
                 mfc=cmap(veh_number + 1),  markersize=16)

        # ax1.plot([s_dep[0]], [s_dep[1]], '.',  markersize=20)

        ax1.quiver(
            lons[:-1],
            lats[:-1],
            lons[1:] - lons[:-1],
            lats[1:] - lats[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=cmap(veh_number + 1)
        )
        # width=0.01)


def plot_vehicle_routes_wrapper(vehicle_routes, customers_coords, starts, ends):
    # Plotting of the routes in matplotlib.
    figsize = (10, 8)

    fig = plt.figure(figsize=figsize)

    graph.set_plot_font()

    ax = fig.add_subplot(111)

    # Plot all the nodes as black dots.
    clon, clat = zip(*[c for c in customers_coords])
    ax.plot(clon, clat, 'k.', markersize=5)
    # ax.plot(clon, clat, 'b.', markersize=20)
    # plot the routes as arrows
    plot_vehicle_routes(vehicle_routes, ax,
                        customers_coords, starts, ends, True)

    plt.grid(zorder=0)
    graph.set_disp("Crowdsensing model - VRP", "longitude", "latitude")

    plt.show()
    return fig
