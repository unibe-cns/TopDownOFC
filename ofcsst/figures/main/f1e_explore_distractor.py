import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter
from ofcsst.simulation.databases import BEST_PARAM_TABLE, get_all_cols
from ofcsst.simulation.scan import get_db_path, find_best_params
from ofcsst.utils import constants, ids, keys, paths
from ofcsst.figures import helper

NAME = "distractor_landscape"
PLOT_DIR = paths.MAIN_FIG_DIR / 'fig1'


def performance_cmap():
    pcm = plt.get_cmap('PuRd')
    cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=pcm.name, a=0.1, b=0.4),
                                             pcm(np.linspace(0.1, 0.4, 101)))
    return cmap


def plot_performance_2d(simulation_type: ids.SimulationType = ids.SIM_PG_NO_GAIN, panel: str = 'e',
                        plot_dir: Path = paths.MAIN_FIG_DIR, save: bool = False) -> None:

    # Preparations
    out_key = keys.PERFORMANCE
    xyz_keys = [keys.SIGNAL_GAIN, keys.NR_DISTRACTOR, out_key]
    table = BEST_PARAM_TABLE
    db_path = get_db_path(scan_type=ids.ST_VARY_DISTRACTION_2D, simulation_type=simulation_type,
                          task_id=constants.TASK_ID, non_stationary=False)

    # Get data
    data_matrix = helper.get_data_by_keys(path=db_path, table=table, key_list=xyz_keys)
    min_v, max_v = 50, 100
    data_matrix[2, data_matrix[2, :] >= max_v] = 0.999 * max_v
    data_matrix[2, data_matrix[2, :] <= min_v] = 1.001 * min_v

    # Plot data
    helper.set_style()
    fig, axs = plt.subplots(figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))
    v = np.linspace(start=min_v, stop=max_v, num=101, endpoint=True)
    cmap = performance_cmap()
    cntr = axs.tricontourf(data_matrix[0, :], data_matrix[1, :], data_matrix[2, :], v, cmap=cmap, extend="neither")

    # Prevent SVG edge shenanigans
    for c in cntr.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.1)

    # Plot additions
    axs.plot(data_matrix[0, :], data_matrix[1, :], 'ko', ms=helper.MARKER_SIZE)
    axs.plot([data_matrix[0, 0], data_matrix[0, -1]], [128, 128], "k--", linewidth=helper.LINE_WIDTH)
    axs.plot([1, 1], [data_matrix[1, 0], data_matrix[1, -1]], "k--", linewidth=helper.LINE_WIDTH)

    # Set axes
    axs.set_xlabel(helper.key_name(key=xyz_keys[0]))
    axs.set_ylabel(helper.key_name(key=xyz_keys[1]))
    axs.set_xscale('log')
    axs.set_yscale('log')
    locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=5)
    axs.xaxis.set_minor_locator(locmin)
    axs.xaxis.set_minor_formatter(NullFormatter())
    axs.set_xticks([0.1, 1., 10.])
    axs.yaxis.set_minor_locator(locmin)
    axs.yaxis.set_minor_formatter(NullFormatter())

    # Colorbar
    cbar = fig.colorbar(cntr, pad=0.03)
    cbar.set_ticks([50, 100])
    cbar.set_label(helper.key_name(key=keys.PERFORMANCE), rotation=270, labelpad=-3.)

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=plot_dir, plot_name=f'{panel}_distractor_landscape')


def scan() -> None:
    """
    Run simulations with pure policy gradient without gain modulation with a variety
    """
    find_best_params(task_id=constants.TASK_ID, scan_type=ids.ST_VARY_DISTRACTION_2D,
                     simulation_type=ids.SIM_PG_NO_GAIN, switch_task=False)


def plot(save: bool = True) -> None:
    plot_performance_2d(simulation_type=ids.SIM_PG_NO_GAIN, panel='e', plot_dir=PLOT_DIR, save=save)
