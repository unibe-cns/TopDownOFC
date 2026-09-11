import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mp_colors
from typing import cast
from ofcsst.utils import constants, paths
from ofcsst.figures import helper

PLOT_DIR = paths.MAIN_FIG_DIR / "fig1"


def plot(save: bool = True) -> None:
    """
    Inset for Fig. 1d showing the effect of apical activity on the gain of the neural input-output function.
    """

    panel = "d"
    helper.set_style()
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(helper.PANEL_WIDTH / 2.5, helper.PANEL_HEIGHT / 2.5),
        squeeze=False,
    )
    fig = cast(plt.Figure, fig)
    ax = ax[0, 0]
    base_colors = ["k", helper.COLOR_GAIN]
    n_gains = 10
    cmap = mp_colors.LinearSegmentedColormap.from_list(
        name="", colors=base_colors, N=n_gains
    )
    colors = cmap(np.linspace(start=0, stop=1, num=n_gains))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mp_colors.Normalize(vmin=0, vmax=1))

    apical_activity = np.linspace(start=0, stop=9, num=n_gains, endpoint=True)
    bottom_up_input = np.linspace(start=-1, stop=1, num=11, endpoint=True)
    for i in range(n_gains):
        gain = min(constants.MAX_GAIN, 1 + max(0.0, apical_activity[i]))
        firing_rates = torch.relu(torch.tensor(gain * bottom_up_input))
        plt.plot(bottom_up_input, firing_rates, color=colors[i])

    # Plotting Specifics
    ymin = -0.05
    ymax = 5
    plt.xlabel(r"$\mathregular{x_{bas}}$", labelpad=0)
    plt.ylabel(r"$\mathregular{z}$", labelpad=-2)
    plt.xticks([-1, 0, 1])
    plt.yticks([0, ymax])
    plt.xlim([bottom_up_input[0], bottom_up_input[-1]])
    plt.ylim([ymin, ymax])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=10)
    cbar.set_ticks([(0.5 + i) / n_gains for i in range(n_gains)])
    ticklabels = np.array(["" for _ in range(n_gains)])
    ticklabels[[0, -1]] = apical_activity.astype(int).astype(str)[[0, -1]]
    cbar.set_ticklabels(ticklabels)
    cbar.ax.yaxis.set_tick_params(pad=2)
    cbar.ax.minorticks_off()
    cbar.set_label(r"$\mathregular{x_{ap}}$", rotation=270, labelpad=4)
    ax.tick_params(axis="both", length=1, pad=1)
    cbar.ax.tick_params(axis="both", length=1, pad=1)

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    plt.legend(frameon=False, bbox_to_anchor=(0.5, 1.1))

    # Save or display
    helper.save_or_show(
        save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=f"{panel}_gain_inset"
    )
