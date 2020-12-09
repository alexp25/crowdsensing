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
FSIZE_LABEL = 16
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

    if not separate:
        fig = plt.figure(id)

    set_plot_font()

    for i, ts in enumerate(timeseries_array):
        if separate:
            fig = plt.figure(id)
        id += 1
        x = xval
        y = ts
        plt.plot(x, y, label=labels[i], color=colors[i])
        # plt.plot(x, y)

        if separate:
            set_disp(title, xlabel, ylabel)
            plt.legend()
            plt.show(block=False)

    if not separate:
        set_disp(title, xlabel, ylabel)
        plt.legend()
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
