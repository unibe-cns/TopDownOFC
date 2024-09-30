import pathlib
import numpy as np
import matplotlib.pyplot as plt
from ofcsst.utils import constants, ids, keys, paths, sql
from ofcsst.simulation import scan, databases
from ofcsst.figures import helper
from ofcsst.figures.main.f1e_explore_distractor import PLOT_DIR


def plot_gain_dependence(simulation_type: ids.SimulationType = ids.SIM_PG_NO_GAIN,
                         plot_dir: pathlib.Path = paths.PLOT_DIR, panel: str = 'f', save: bool = True) -> None:

    # Preparations
    cols = [f"{keys.PERFORMANCE}_{n}" for n in range(constants.NR_TRIALS - constants.PERF_MAV_N, constants.NR_TRIALS)]
    db_path = scan.get_db_path(scan_type=ids.ST_VARY_DISTRACTION_2D, simulation_type=simulation_type,
                               task_id=constants.TASK_ID, non_stationary=False)
    conn = sql.connect(db_path=db_path)
    cur = conn.cursor()

    # Get data
    snr = scan.SIGNAL_AMPLITUDES
    performance_av = np.zeros((len(scan.SIGNAL_AMPLITUDES),))
    performance_se = np.zeros((len(scan.SIGNAL_AMPLITUDES),))
    for n in range(len(scan.SIGNAL_AMPLITUDES)):
        where_values = {keys.NR_DISTRACTOR: constants.NUMBER_DISTRACTOR,
                        keys.SIGNAL_GAIN: scan.SIGNAL_AMPLITUDES[n]}
        best_lr = databases.get_best_lr(simulation_type=simulation_type, unique_values=where_values, cur=cur)
        where_values[keys.LEARNING_RATE_PG] = best_lr[0]
        if simulation_type == ids.SIM_CONVEX_NO_GAIN:
            where_values[keys.LEARNING_RATE_Q] = best_lr[1]
        performances = sql.select_where(
            db_path=db_path, table=databases.DATA_TABLE, get_cols=cols, where_values=where_values)
        performances = np.mean(np.array(performances), axis=1)
        performance_av[n] = np.mean(performances)
        performance_se[n] = np.std(performances)

    # Plot data
    helper.set_style()
    fig = plt.figure(figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))
    plt.plot(snr, performance_av, color=helper.COLOR_NO_GAIN, marker="o", ms=helper.MARKER_SIZE, mfc="k", mec="k",
             clip_on=False, label='Without gain\nadaptation')
    plt.fill_between(snr, performance_av - performance_se, performance_av + performance_se,
                     color=helper.COLOR_NO_GAIN, alpha=0.3, linewidth=0., clip_on=False)

    # Plotting Specifics
    ymin = 0.4
    ymax = 1.
    plt.xlabel(helper.key_name(key=keys.SIGNAL_GAIN))
    plt.ylabel(helper.key_name(key=keys.PERFORMANCE))
    plt.xscale("log")
    plt.xticks([0.01, 0.1, 1., 10., 100.])
    plt.yticks([0.4, 0.6, 0.8, 1.], labels=["40", "60", "80", "100"])
    plt.xlim([snr[0], 1.06 * snr[-1]])
    plt.ylim([ymin, ymax])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds([ymin, 1.])

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    helper.set_panel_label(label=panel, fig=fig)
    plt.legend(frameon=False, bbox_to_anchor=(0.5, 1.1))

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=plot_dir, plot_name=f"{panel}_vary_snr")


def plot(save: bool = True) -> None:
    plot_gain_dependence(simulation_type=ids.SIM_PG_NO_GAIN, plot_dir=PLOT_DIR, panel='f', save=save)
