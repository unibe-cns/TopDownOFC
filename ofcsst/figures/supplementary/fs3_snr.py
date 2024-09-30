import numpy as np
import matplotlib.pyplot as plt
from ofcsst.simulation import scan, databases
from ofcsst.utils import constants, ids, keys, paths
from ofcsst.figures import helper
from ofcsst.figures.main.f1e_explore_distractor import performance_cmap

NAME = "snr"
PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / 'figs3'


def panel_a_2d(save: bool = False) -> None:

    # Preparations
    simulation_type = ids.SIM_PG_NO_GAIN
    out_key = keys.PERFORMANCE
    xyz_keys = [keys.SNR, keys.NR_DISTRACTOR, out_key]
    table = databases.BEST_PARAM_TABLE
    db_path = scan.get_db_path(scan_type=ids.ST_VARY_DISTRACTION_2D, simulation_type=simulation_type,
                               task_id=constants.TASK_ID, non_stationary=False)

    # Get data
    data_matrix = helper.get_data_by_keys(path=db_path, table=table, key_list=xyz_keys)
    min_v, max_v = 50, 100
    data_matrix[2, data_matrix[2, :] >= max_v] = 0.999 * max_v
    data_matrix[2, data_matrix[2, :] <= min_v] = 1.001 * min_v

    # Plot data
    helper.set_style()
    fig, axs = plt.subplots(figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))
    axs.set_facecolor((0.6, 0.6, 0.6))
    v = np.linspace(min_v, max_v, 101, endpoint=True)
    cmap = performance_cmap()
    cntr = axs.tricontourf(data_matrix[0, :], data_matrix[1, :], data_matrix[2, :], v, cmap=cmap, extend="neither")
    axs.plot(data_matrix[0, :], data_matrix[1, :], 'ko', ms=helper.MARKER_SIZE)

    # Set axes
    axs.set_xlabel(helper.key_name(key=xyz_keys[0]))
    axs.set_ylabel(helper.key_name(key=xyz_keys[1]))
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xticks([10**x for x in range(-3, 3)])
    axs.set_xticks([10**x for x in range(-5, 3)],
                   ['' if x % 2 else fr'$\mathregular{{10^{{{x}}}}}$' for x in range(-5, 3)])

    # Color bar
    cbar = fig.colorbar(cntr, pad=0.03)
    cbar.set_ticks([50, 100])
    cbar.set_label("Performance", rotation=270, labelpad=-3.)

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    helper.set_panel_label(label="a", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=f'a_{NAME}_2d', transparent_background=False)


def panel_b_1d(save: bool = False) -> None:

    # Preparations
    simulation_type = ids.SIM_PG_NO_GAIN
    x_key = keys.SNR
    xy_keys = [x_key, keys.PERFORMANCE]
    table = databases.BEST_PARAM_TABLE
    db_path = scan.get_db_path(scan_type=ids.ST_VARY_DISTRACTION_2D, simulation_type=simulation_type,
                               task_id=constants.TASK_ID, non_stationary=False)

    # Get data
    data_matrix = helper.get_data_by_keys(path=db_path, table=table, key_list=xy_keys)
    min_v, max_v = 50, 100
    data_matrix[1, data_matrix[1, :] >= max_v] = 0.999 * max_v
    data_matrix[1, data_matrix[1, :] <= min_v] = 1.001 * min_v

    # Plot data
    helper.set_style()
    fig, axs = plt.subplots(figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))
    axs.plot(data_matrix[0, :], data_matrix[1, :], 'ko', ms=helper.MARKER_SIZE)

    # Set axes
    axs.set_xlabel(helper.key_name(key=xy_keys[0]))
    axs.set_ylabel(helper.key_name(key=xy_keys[1]))
    axs.set_xscale('log')
    axs.set_ylim([45, 100])
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xticks([10**x for x in range(-6, 4)],
                   ['' if x % 2 else fr'$\mathregular{{10^{{{x}}}}}$' for x in range(-6, 4)])

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    helper.set_panel_label(label="b", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=f'b_{NAME}_1d')


def plot(save: bool = True):
    panel_a_2d(save=save)
    panel_b_1d(save=save)
