import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_and_show_figure(
    savepath,
    fig,
    save_only: bool = False,
    show: bool = True,
    dpi_save: int = 300,
    background_color: Optional[str] = None,
    tight: bool = True,
    verbose: bool = False
    ):

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