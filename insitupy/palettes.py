
import matplotlib as mpl
import matplotlib.cm as cm
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
        self.tab20_mod = ListedColormap([rgb2hex(elem) for elem in color_cycle], name="tab20_mod")
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

def create_colormap(
    N,
    colormaps = [cm.Reds_r, cm.Blues_r, cm.Greens_r, cm.Purples_r, cm.Greys_r]
    ):
    """
    Adapted from: https://stackoverflow.com/questions/72171993/how-to-extend-the-color-palette-in-matplotlib
    """
    # extract the following number of colors for each colormap
    n_cols_per_cm = int(np.ceil(N / len(colormaps)))
    # discretize the colormap. Note the upper limit of 0.75, so we
    # avoid too white-ish colors
    discr = np.linspace(0, 0.75, n_cols_per_cm)

    # extract the colors
    colors = np.zeros((n_cols_per_cm * len(colormaps), 4))
    for i, cmap in enumerate(colormaps):
        colors[i * n_cols_per_cm : (i + 1) * n_cols_per_cm, :] = cmap(discr)

    # convert to hex
    colors_hex = [rgb2hex(elem) for elem in colors]
    return colors_hex


def cmap2hex(cmap):
    '''
    Generate list of hex-coded colors from cmap.
    '''
    hexlist = [rgb2hex(cmap(i)) for i in range(cmap.N)]
    return hexlist

# --- Geometry and transcript colour palettes ---
# Generated with glasbey (not a runtime dependency).
# To regenerate:
#   import glasbey
#   ANNOTATIONS_PALETTE = glasbey.create_palette(palette_size=64, colorblind_safe=True)
#   REGIONS_PALETTE = glasbey.create_palette(palette_size=64, colorblind_safe=True,
#                         lightness_bounds=(20, 40), chroma_bounds=(40, 50))
#   TRANSCRIPTS_PALETTE = glasbey.create_palette(palette_size=256, colorblind_safe=True)

ANNOTATIONS_PALETTE: list = [
    "#0c71ff", "#ca2800", "#ff28ba", "#000096", "#59d700", "#1c5951", "#20d2ff", "#69a28a",
    "#650000", "#5d0486", "#b20065", "#ffaa96", "#ba10c2", "#510039", "#00650c", "#0096a6",
    "#71a600", "#ff96db", "#ff316d", "#0018ff", "#00ffdb", "#fb3dff", "#ff5114", "#8e0069",
    "#24ffa6", "#002d1c", "#8e7565", "#042441", "#00658a", "#c69aaa", "#922020", "#967d8e",
    "#599aff", "#69c66d", "#7d043d", "#f7eb82", "#a6a2c2", "#008671", "#9210ce", "#beebff",
    "#ca0c96", "#390c9a", "#413d59", "#ba8e69", "#f7c6d7", "#004100", "#657900", "#e71486",
    "#00beb2", "#ce1c45", "#b26db6", "#b639ff", "#ffe3c2", "#f3b255", "#490018", "#71595d",
    "#8acaae", "#004db6", "#494531", "#dbaaff", "#8a0892", "#795d82", "#009e51", "#a6f700",
]

REGIONS_PALETTE: list = [
    "#616dbe", "#a26504", "#792049", "#008a75", "#005108", "#612475", "#ae518e", "#aa4d55",
    "#107135", "#862d28", "#86458e", "#005996", "#007d8a", "#755100", "#792465", "#9a3d65",
    "#a64d2d", "#9a5daa", "#005d2d", "#148e5d", "#5d8639", "#8e4175", "#6d4da2", "#92414d",
    "#49318a", "#ae5575", "#693100", "#006986", "#6d2435", "#656d00", "#246518", "#2086aa",
    "#247d51", "#753179", "#965592", "#8a4524", "#048a86", "#084d75", "#59418a", "#2469a6",
    "#862d61", "#b25949", "#711851", "#965504", "#9e3d3d", "#7d59a2", "#a64979", "#692065",
    "#863551", "#823508", "#a66531", "#24759a", "#a64561", "#7d3d75", "#495100", "#753d8e",
    "#717d14", "#86283d", "#6d281c", "#28823d", "#048665", "#516d24", "#8a65be", "#006d45",
]

TRANSCRIPTS_PALETTE: list = [
    "#0c71ff", "#ca2800", "#ff28ba", "#000096", "#86e300", "#005d55", "#20d2ff", "#20ae86",
    "#590000", "#65008e", "#b6005d", "#ffaa96", "#ba10c2", "#510039", "#00650c", "#0096a6",
    "#20aa00", "#ffaeeb", "#ff316d", "#0431ff", "#45e7d7", "#df6dff", "#ff6d2d", "#8a2071",
    "#24ffa6", "#002d1c", "#7d7151", "#042441", "#28658a", "#c69aaa", "#922020", "#927186",
    "#599aff", "#69c66d", "#6d2d41", "#ffe779", "#a2a2c6", "#008671", "#9204d2", "#beebff",
    "#c6149a", "#3d10a6", "#413d59", "#b28a5d", "#ebcad2", "#004100", "#657900", "#e71486",
    "#00beb2", "#ce1c45", "#a671b2", "#b639ff", "#fbe7c2", "#f3b255", "#490018", "#71595d",
    "#9ac6aa", "#144dbe", "#414935", "#dbaaff", "#8e049a", "#82598a", "#009e51", "#ceff00",
    "#fb75df", "#397dbe", "#aeae00", "#b2797d", "#ba7504", "#b6beca", "#f33539", "#c2f3df",
    "#4d0051", "#0000e7", "#799696", "#003939", "#ff55a2", "#ef20d2", "#396d71", "#ff7992",
    "#10b2d7", "#864d04", "#081869", "#ba82aa", "#453120", "#613500", "#00754d", "#df18f3",
    "#9a4575", "#35d7db", "#fb8269", "#ff82c6", "#5d4959", "#a6a2ff", "#71006d", "#49eb6d",
    "#ffdbf7", "#004d79", "#790051", "#a2a28a", "#aa5565", "#412d35", "#9e82db", "#41046d",
    "#084535", "#b26139", "#202400", "#75001c", "#5d8e71", "#924545", "#0855ff", "#ffa6be",
    "#e7d28e", "#7955aa", "#ffd21c", "#2d2035", "#96a241", "#0069be", "#aa009a", "#ca519a",
    "#614975", "#69798e", "#aed2ff", "#28fff3", "#185965", "#00ebb6", "#9a005d", "#188239",
    "#92aea6", "#bedbb2", "#b61820", "#96043d", "#5d6131", "#bedfdf", "#79ff5d", "#18d724",
    "#00ceae", "#61ae6d", "#5d459a", "#9665c2", "#59a6b2", "#ca5d5d", "#8e00ff", "#ce28ba",
    "#00ca8e", "#ca7dce", "#c2ae82", "#d2a2ce", "#597df7", "#08395d", "#df0065", "#009aca",
    "#087d96", "#ff920c", "#ca0882", "#188e8a", "#ca6992", "#c6fba6", "#3d695d", "#28e392",
    "#650035", "#00efff", "#716171", "#5914aa", "#552d49", "#758e2d", "#868aa2", "#ae08db",
    "#d2a255", "#e7b2ce", "#96bedf", "#009e86", "#7d55e7", "#d7aaa2", "#2d1c20", "#3900d2",
    "#657569", "#e37571", "#ebceff", "#a26196", "#9ea2ae", "#925931", "#f3490c", "#245118",
    "#314549", "#e700b2", "#713531", "#fb7dff", "#593575", "#db7992", "#ffc2ba", "#86968a",
    "#d2793d", "#a20000", "#868655", "#7182ae", "#41beca", "#24b6ff", "#9e8692", "#b2557d",
    "#a2d7e3", "#aa1082", "#aaffc6", "#00397d", "#044520", "#718282", "#b2ca82", "#ffdfdb",
    "#aadbc6", "#d761ba", "#5500d2", "#8296c2", "#a6650c", "#fb2d86", "#2dbe31", "#8e3955",
    "#ca8a00", "#71cac2", "#cebae7", "#41180c", "#2800a6", "#dfbe1c", "#ba0849", "#511869",
    "#be5dff", "#694520", "#ff96f3", "#ca61d7", "#008a59", "#711cbe", "#dfcae3", "#187500",
    "#653561", "#e771b2", "#820008", "#b2fbf3", "#655d49", "#a255aa", "#fb24a2", "#797935",
    "#ff5555", "#c29ae7", "#fba271", "#db1c2d", "#754d71", "#614949", "#3dae9a", "#18ae49",
    "#412d04", "#ff7dae", "#18ffdf", "#610051", "#aac251", "#dbd761", "#085d45", "#fb55eb",
]


def map_to_colors(cat_list, palette):
    """Map a list of categories to hex colour strings using a matplotlib colormap.

    Cycles through *palette* using modular indexing, so the palette wraps
    around if there are more categories than colours.

    Args:
        cat_list: Sequence of category labels to assign colours to.
        palette: A matplotlib :class:`~matplotlib.colors.Colormap` with a
            ``N`` attribute (number of colours).

    Returns:
        A ``dict`` mapping each category label to its hex colour string.
    """
    color_dict = {cat: rgb2hex(palette(i % palette.N)) for i, cat in enumerate(cat_list)}
    return color_dict
