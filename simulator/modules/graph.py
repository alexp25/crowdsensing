import matplotlib.pylab as plt
import matplotlib
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import numpy as np


# FSIZE_TITLE = 16
# FSIZE_LABEL = 14
# FSIZE_LABEL_S = 14
# FSIZE_LABEL_XS = 12


# FSIZE_TITLE = 16
# FSIZE_LABEL = 14
# FSIZE_LABEL_S = 14
# FSIZE_LABEL_XS = 12
# OPACITY = 0.9

FSIZE_TITLE = 18
FSIZE_LABEL = 18
FSIZE_LABEL_S = 16
FSIZE_LABEL_XS = 14
OPACITY = 0.9


def set_plot_font(size=FSIZE_LABEL_XS):
    plt.rc('xtick', labelsize=size)
    plt.rc('ytick', labelsize=size)
    plt.rc('legend', fontsize=size)


def plot_timeseries_multi(timeseries_array, xval, labels, colors, title, xlabel, ylabel, separate):
    matplotlib.style.use('default')
    id = 0

    fig = None

    figsize = (10,8)

    if not separate:
        fig = plt.figure(id, figsize=figsize)

    set_plot_font()

    plt.grid(zorder=0)
    
    legend_loc = "upper left"

    for i, ts in enumerate(timeseries_array):
        if separate:
            fig = plt.figure(id, figsize=figsize)
        id += 1
        x = xval
        y = ts
        plt.plot(x, y, label=labels[i], color=colors[i], linewidth=3)
        # plt.plot(x, y)

        if separate:
            set_disp(title, xlabel, ylabel)
            plt.grid(zorder=0)
            plt.legend(loc=legend_loc, fontsize=FSIZE_LABEL_S)
            plt.show(block=False)

    # plt.xticks(rotation=rotation)

    plt.grid(zorder=0) 

    if not separate:
        set_disp(title, xlabel, ylabel)
        plt.legend(loc=legend_loc, fontsize=FSIZE_LABEL_S)
        fig = plt.gcf()
        plt.show()

    if separate:
        plt.show()

    return fig

def save_figure(fig, file):
    fig.savefig(file, dpi=300)


def set_disp_ax(ax, title, xlabel, ylabel):
    if title:
        ax.set_title(title,  fontsize=FSIZE_TITLE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FSIZE_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FSIZE_LABEL)


def set_disp(title, xlabel, ylabel):
    if title:
        plt.gca().set_title(title, fontsize=FSIZE_TITLE)
    if xlabel:
        plt.xlabel(xlabel, fontsize=FSIZE_LABEL)
    if ylabel:
        plt.ylabel(ylabel, fontsize=FSIZE_LABEL)
