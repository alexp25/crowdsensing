
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from modules import graph
from svgpath2mpl import parse_path
from xml.dom import minidom

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
    # cmap = discrete_cmap(len(data)+2, 'hsv')
    cmap = discrete_cmap(len(data)+2, 'nipy_spectral')
    # cmap = discrete_cmap(len(data) + 2, 'jet')
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


def plot_coords_map(coords):
    clat, clon = zip(*[c for c in coords])
    plt.plot(clat, clon)
    plt.show()


def plot_coords_map_multi(coords_vect):
    figsize = (10, 8)

    fig = plt.figure(figsize=figsize)

    graph.set_plot_font()

    ax = fig.add_subplot(111)

    colors = create_discrete_cmap(coords_vect)
    colors_map = [colors(i+1) for i in range(len(coords_vect))]
    for i, coords in enumerate(coords_vect):
        clat, clon = zip(*[c for c in coords])
        ax.plot(clon, clat, color=colors_map[i])

    plt.grid(zorder=0)
    graph.set_disp("Random walk", "longitude", "latitude")
    plt.show()
    return fig


def plot_vehicle_routes_wrapper(vehicle_routes, place_coords, item_coords, colors, starts, ends, zoom_out):
    # Plotting of the routes in matplotlib.
    figsize = (10, 8)

    fig = plt.figure(figsize=figsize)

    graph.set_plot_font()

    ax = fig.add_subplot(111)

    icon_path = "iconmonstr-location-1.svg"
    doc = minidom.parse(icon_path)  # parseString also exists
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]

    smiley = parse_path("""m 739.01202,391.98936 c 13,26 13,57 9,85 -6,27 -18,52 -35,68 -21,20 -50,23 -77,18 -15,-4 -28,-12 -39,-23 -18,-17 -30,-40 -36,-67 -4,-20 -4,-41 0,-60 l 6,-21 z m -302,-1 c 2,3 6,20 7,29 5,28 1,57 -11,83 -15,30 -41,52 -72,60 -29,7 -57,0 -82,-15 -26,-17 -45,-49 -50,-82 -2,-12 -2,-33 0,-45 1,-10 5,-26 8,-30 z M 487.15488,66.132209 c 121,21 194,115.000001 212,233.000001 l 0,8 25,1 1,18 -481,0 c -6,-13 -10,-27 -13,-41 -13,-94 38,-146 114,-193.000001 45,-23 93,-29 142,-26 z m -47,18 c -52,6 -98,28.000001 -138,62.000001 -28,25 -46,56 -51,87 -4,20 -1,57 5,70 l 423,1 c 2,-56 -39,-118 -74,-157 -31,-34 -72,-54.000001 -116,-63.000001 -11,-2 -38,-2 -49,0 z m 138,324.000001 c -5,6 -6,40 -2,58 3,16 4,16 10,10 14,-14 38,-14 52,0 15,18 12,41 -6,55 -3,3 -5,5 -5,6 1,4 22,8 34,7 42,-4 57.6,-40 66.2,-77 3,-17 1,-53 -4,-59 l -145.2,0 z m -331,-1 c -4,5 -5,34 -4,50 2,14 6,24 8,24 1,0 3,-2 6,-5 17,-17 47,-13 58,9 7,16 4,31 -8,43 -4,4 -7,8 -7,9 0,0 4,2 8,3 51,17 105,-20 115,-80 3,-15 0,-43 -3,-53 z m 61,-266 c 0,0 46,-40 105,-53.000001 66,-15 114,7 114,7 0,0 -14,76.000001 -93,95.000001 -76,18 -126,-49 -126,-49 z""")
    smiley = parse_path(path_strings[0])
    smiley.vertices -= smiley.vertices.mean(axis=0)

    marker = smiley
    marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))

    markerface = ['y.', 'g.', 'b.']
    for i, item in enumerate(item_coords):
        clat, clon = zip(*[c for c in item_coords[item]])
        # markerface[i % len(colors)]
        ax.plot(clon, clat, '.', color=colors[i], markersize=10)

    # Plot all the nodes as black dots.
    clat, clon = zip(*[c for c in place_coords])
    ax.plot(clon, clat, 'm.', color=colors[len(
        colors)-1], marker=marker, markersize=10)
    # ax.plot(clon, clat, 'b.', markersize=20)

    # ax.plot(clon, clat, 'b.', markersize=20)
    # plot the routes as arrows

    ax.legend(["T", "C", "S", "M", "P"])

    if vehicle_routes is not None:
        plot_vehicle_routes(vehicle_routes, ax,
                            place_coords, starts, ends, True)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # example of how to zoomout by a factor of 0.1
    factor = zoom_out
    new_xlim = (xlim[0] + xlim[1])/2 + np.array((-0.5, 0.5)) * \
        (xlim[1] - xlim[0]) * (1 + factor)
    ax.set_xlim(new_xlim)
    new_ylim = (ylim[0] + ylim[1])/2 + np.array((-0.5, 0.5)) * \
        (ylim[1] - ylim[0]) * (1 + factor)
    ax.set_ylim(new_ylim)

    plt.grid(zorder=0)
    graph.set_disp("DMS fill", "longitude", "latitude")

    plt.show()
    return fig
