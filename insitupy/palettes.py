from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, rgb2hex


class CustomPalettes:
    '''
    Class containing a collection of custom color palettes.
    '''
    def __init__(self):
        # palette for colorblind people. From: https://gist.github.com/thriveth/8560036
        self.colorblind = ListedColormap(
            ['#377eb8', '#ff7f00', '#4daf4a',
             '#f781bf', '#dede00', '#a65628',
             '#984ea3', '#999999', '#e41a1c'], name="colorblind")

        # palette from Caro. Optimized for colorblind people.
        self.caro = ListedColormap(['#3288BD','#440055', '#D35F5F', '#A02C2C','#225500', '#66C2A5', '#447C69'], name="caro")

        # from https://thenode.biologists.com/data-visualization-with-flying-colors/research/
        self.okabe_ito = ListedColormap(["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"], name="okabe_ito")
        self.tol_bright = ListedColormap(["#EE6677", "#228833", "#4477AA", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"], name="tol_bright")
        self.tol_muted = ListedColormap(["#88CCEE", "#44AA99", "#117733", "#332288", "#DDCC77", "#999933", "#CC6677", "#882255", "#AA4499", "#DDDDDD"], name="tol_muted")
        self.tol_light = ListedColormap(["#BBCC33", "#AAAA00", "#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#44BB99", "#DDDDDD"], name="tol_light")

        # generate modified tab20 color palette
        colormap = mpl.colormaps["tab20"]

        # split by high intensity and low intensity colors in tab20
        cmap1 = colormap.colors[::2]
        cmap2 = colormap.colors[1::2]

        # concatenate color cycle
        color_cycle = cmap1[:7] + cmap1[8:] + cmap2[:7] + cmap2[8:] + (cmap1[7],) + (cmap2[7],)
        self.tab20_mod = ListedColormap([rgb2hex(elem) for elem in color_cycle])


    def show_all(self):
        '''
        Plots all colormaps in the collection.
        '''
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # get list of names and respective
        cmaps = []
        names = []
        for name, cmap in vars(self).items():
            if isinstance(cmap, ListedColormap):
                cmaps.append(cmap)
                names.append(name)

        # Create figure and adjust figure height to number of colormaps
        nrows = len(vars(self).values())
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
        fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                            left=0.2, right=0.99)

        axs = axs.ravel()

        for ax, name, cmap in zip(axs, names, cmaps):
            ax.imshow(gradient, aspect='auto', cmap=cmap)
            ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                    transform=ax.transAxes)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
            ax.set_axis_off()


def cmap2hex(cmap):
    '''
    Generate list of hex-coded colors from cmap.
    '''
    hexlist = [rgb2hex(cmap(i)) for i in range(cmap.N)]
    return hexlist

