import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_and_show_figure(
    savepath,
    fig,
    save_only: bool = False,
    show: bool = True,
    dpi_save: int = 300,
    background_color: str | None = None,
    tight: bool = True,
    verbose: bool = False
    ):
    """Save a matplotlib figure to disk and optionally display or close it.

    If *savepath* is not None the parent directory is created automatically.
    The figure is always saved with ``bbox_inches='tight'``.

    Args:
        savepath: File path for the saved figure.  Supported formats are
            those accepted by :func:`matplotlib.pyplot.savefig` (e.g.
            ``".png"``, ``".pdf"``).  If None, no file is written.
        fig: The :class:`matplotlib.figure.Figure` to save or show.
        save_only: If True, close the figure after saving without
            displaying it.  Overrides *show*.  Defaults to False.
        show: If True (and *save_only* is False), call
            :func:`matplotlib.pyplot.show`.  Defaults to True.
        dpi_save: Resolution in dots per inch for the saved image.
            Defaults to 300.
        background_color: Face-color passed to :func:`~matplotlib.pyplot.savefig`
            (e.g. ``"white"`` or ``"#ffffff"``).  If None, the figure's
            current background is used.
        tight: If True, call :meth:`~matplotlib.figure.Figure.tight_layout`
            before saving.  Defaults to True.
        verbose: If True, log the save path.  Defaults to False.
    """
    if tight:
        fig.tight_layout()

    if savepath is not None:
        logger.info("Saving figure to file " + str(savepath)) if verbose else None

        # create path if it does not exist
        Path(os.path.dirname(savepath)).mkdir(parents=True, exist_ok=True)

        # save figure
        plt.savefig(savepath, dpi=dpi_save,
                    facecolor=background_color, bbox_inches='tight')
        logger.info("Saved.") if verbose else None
    if save_only:
        plt.close(fig)
    elif show:
        plt.show()
    else:
        return
