"""save_and_show_figure saves the figure it is given, not the current one."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from insitupy.plotting.save import save_and_show_figure


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_save_uses_given_figure(tmp_path):
    fig1 = plt.figure(figsize=(8, 2))  # wide
    fig1.add_subplot()
    fig2 = plt.figure(figsize=(3, 3))  # square, and the current figure
    fig2.add_subplot()
    assert plt.gcf() is fig2

    path = tmp_path / "a.png"
    save_and_show_figure(savepath=path, fig=fig1, save_only=True, tight=False, dpi_save=50)

    img = plt.imread(path)
    height, width = img.shape[:2]
    assert width / height > 2  # the wide figure, not the square current one


def test_save_directory_path_raises(tmp_path):
    fig = plt.figure()
    fig.add_subplot()
    with pytest.raises(ValueError, match="file path"):
        save_and_show_figure(savepath=str(tmp_path / "plots") + "/", fig=fig, save_only=True)
    with pytest.raises(ValueError, match="file path"):
        save_and_show_figure(savepath=tmp_path, fig=fig, save_only=True)
