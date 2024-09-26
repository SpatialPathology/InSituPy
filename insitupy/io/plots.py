import os
from pathlib import Path

import matplotlib.pyplot as plt


def save_and_show_figure(savepath, fig, save_only=False, show=True, dpi_save=300, background_color=None, tight=True):

    if tight:
        fig.tight_layout()

    if savepath is not None:
        print("Saving figure to file " + savepath)

        # create path if it does not exist
        Path(os.path.dirname(savepath)).mkdir(parents=True, exist_ok=True)

        # save figure
        plt.savefig(savepath, dpi=dpi_save,
                    facecolor=background_color, bbox_inches='tight')
        print("Saved.")
    if save_only:
        plt.close(fig)
    elif show:
        return plt.show()
    else:
        return