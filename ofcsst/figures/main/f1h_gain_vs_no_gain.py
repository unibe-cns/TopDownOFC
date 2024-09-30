import pathlib
import numpy as np
import matplotlib.pyplot as plt
from ofcsst.simulation import databases
from ofcsst.simulation.scan import get_db_path, find_best_params, NUMBER_DISTRACTORS
from ofcsst.utils import ids, keys, paths, sql, constants
from ofcsst.figures import helper
from ofcsst.figures.main.f1e_explore_distractor import PLOT_DIR


def plot_gain_vs_no_gain(simulation_types: tuple[ids.SimulationType, ids.SimulationType],
                         plot_dir: pathlib.Path = paths.PLOT_DIR, panel: str = 'h', save: bool = True) -> None:

    # Plotting preparations
    xy_keys = [keys.NR_DISTRACTOR, keys.PERFORMANCE]
    cols = [f"{keys.PERFORMANCE}_{n}" for n in range(constants.NR_TRIALS - constants.PERF_MAV_N, constants.NR_TRIALS)]
    labels = ["Without gain adaptation", "With gain adaptation"]
    nr_sims = len(simulation_types)
    assert nr_sims == 2
    colors = [helper.COLOR_NO_GAIN, helper.COLOR_GAIN, "r", "b"]
    plot_args = [{"marker": "o", "ms": helper.MARKER_SIZE, "mfc": "k", "mec": "k"}, {}, {}, {}]
    for c in range(len(simulation_types)):
        plot_args[c]["label"] = labels[c]
        plot_args[c]["color"] = colors[c]
    scan_types = [ids.ST_VARY_DISTRACTION_2D, ids.ST_VARY_DISTRACTOR_1D, ids.ST_VARY_DISTRACTION_2D,
                  ids.ST_VARY_DISTRACTOR_1D]
    db_paths = [get_db_path(scan_type=scan_types[i], simulation_type=simulation_types[i],
                            task_id=constants.TASK_ID, non_stationary=False) for i in range(nr_sims)]

    # plot
    helper.set_style()
    fig = plt.figure(figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))
    for condition in [1, 0]:
        td = simulation_types[condition] in ids.TD_LR_SIMULATIONS
        convex = simulation_types[condition] in [ids.SIM_CONVEX_NO_GAIN, ids.SIM_CONVEX_GAIN]
        cur = sql.connect(db_path=db_paths[condition]).cursor()
        performance_av = np.zeros((len(NUMBER_DISTRACTORS),))
        performance_se = np.zeros((len(NUMBER_DISTRACTORS),))
        for n in range(len(NUMBER_DISTRACTORS)):

            # Get parameters
            where_values = {keys.NR_DISTRACTOR: NUMBER_DISTRACTORS[n], keys.SIGNAL_GAIN: 1.}
            best_lr = databases.get_best_lr(simulation_type=simulation_types[condition], unique_values=where_values,
                                            cur=cur)
            if convex:
                where_values[keys.LEARNING_RATE_PG] = best_lr[0]
                where_values[keys.LEARNING_RATE_Q] = best_lr[1]
                if td:
                    where_values[keys.LEARNING_RATE_V] = best_lr[2]
            else:
                where_values[keys.LEARNING_RATE_PG] = best_lr[0]
                if td:
                    where_values[keys.LEARNING_RATE_V] = best_lr[1]

            # Get performances
            performances = sql.select_where(db_path=db_paths[condition], table=databases.DATA_TABLE,
                                            get_cols=cols, where_values=where_values)
            performances = np.mean(np.array(performances), axis=0)
            performance_av[n] = np.mean(performances)
            performance_se[n] = np.std(performances)
        # print(condition, n, performance_av)
        plt.plot(NUMBER_DISTRACTORS, performance_av, **plot_args[condition], clip_on=False)
        plt.fill_between(NUMBER_DISTRACTORS, performance_av - performance_se,
                         performance_av + performance_se, color=colors[condition], alpha=0.3, linewidth=0.,
                         clip_on=False)

    # axes and axis labels
    plt.xlabel(helper.key_name(key=xy_keys[0]))
    plt.ylabel(helper.key_name(key=xy_keys[1]))
    plt.legend(frameon=False)
    plt.xscale("log")
    plt.yticks([0.4, 0.6, 0.8, 1.], labels=["40", "60", "80", "100"])
    plt.xlim([NUMBER_DISTRACTORS[0], 1.06 * NUMBER_DISTRACTORS[-1]])
    plt.ylim([0.4, 1.])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=plot_dir, plot_name=f"{panel}_distractor_dependence")


def plot(save: bool = True):
    plot_gain_vs_no_gain(simulation_types=(ids.SIM_PG_NO_GAIN, ids.SIM_PG_GAIN), plot_dir=PLOT_DIR, panel='h',
                         save=save)


def scan():
    find_best_params(simulation_type=ids.SIM_PG_GAIN, task_id=constants.TASK_ID, switch_task=False,
                     scan_type=ids.ST_VARY_DISTRACTOR_1D)
