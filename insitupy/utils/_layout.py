import math


def get_nrows_maxcols(n_keys, max_cols):
    '''
    Determine optimal number of rows and columns for `plt.subplot` based on
    number of keys ['n_keys'] and maximum number of columns [`max_cols`].

    Returns: `n_plots`, `n_rows`, `max_cols`
    '''

    #n_plots = len(keys)
    if n_keys > max_cols:
        n_rows = math.ceil(n_keys / max_cols)
    else:
        n_rows = 1
        max_cols = n_keys

    return n_keys, n_rows, max_cols

def remove_empty_subplots(axes, nplots, nrows, ncols):
    """Hide trailing empty axes in a flat subplot array.

    After populating *nplots* subplots in a grid of *nrows* × *ncols*, any
    remaining axes beyond *nplots* are hidden with :meth:`set_axis_off`.

    Args:
        axes: 1-D array of :class:`~matplotlib.axes.Axes` objects.
        nplots: Number of subplots that were actually populated.
        nrows: Total number of rows in the subplot grid.
        ncols: Total number of columns in the subplot grid.

    Raises:
        ValueError: If *axes* is not 1-D.
    """
    if len(axes.shape) != 1:
        raise ValueError("Axis object must have only one dimension.")
    if nplots > 1:
        # check if there are empty plots remaining
        i = nplots
        while i < nrows * ncols:
            # remove empty plots
            axes[i].set_axis_off()
            i+=1
