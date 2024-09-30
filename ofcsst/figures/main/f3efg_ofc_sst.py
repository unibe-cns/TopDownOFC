import sqlite3
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import List
from pathlib import Path
import ofcsst.utils.process
from ofcsst import simulation
from ofcsst.utils import ids, keys, paths, sql, constants
from ofcsst.figures import helper

TASK_ID = ids.BINARY_2VN
NAME = "fig_3efg_ofc_sst"
PLOT_DIR = paths.MAIN_FIG_DIR / "fig3"
TRIAL_NRS = (constants.NR_TRIALS, 2 * constants.NR_TRIALS)
OFC_SIM = ids.SIM_CONVEX_OFC
NO_OFC_SIM = ids.SIM_CONVEX_GAIN
SIM_NAMES = {OFC_SIM: 'Intact lOFC', NO_OFC_SIM: 'Silenced lOFC'}
OFC_SCAN = ids.ST_TD
COLOR_WT = helper.COLOR_DEFAULT
COLOR_NO_OFC = helper.COLOR_SST
SAVE_PATH = paths.RESULT_DIR / paths.SIMULATION_SUBDIR / f"{NAME}.db"


def final_simulation(sim_type: ids.SimulationType, scan_path: Path, seeds: [int],
                     save_db_path: Path = SAVE_PATH) -> None:

    # Initializations to save final simulation outcomes
    save_keys = [str(keys.SEED)] + [f"{keys.PERFORMANCE}_{t}" for t in range(sum(TRIAL_NRS))]
    conn = sql.connect(db_path=save_db_path)
    sql.drop_table(conn=conn, table_name=sim_type, verbose=False)
    sql.make_table(conn=conn, table_name=sim_type, col_names=save_keys, verbose=False)
    conn.close()
    insert_cmd = sql.get_insert_cmd(table=sim_type, col_keys=save_keys)

    select_cols = simulation.databases.get_unique_cols(simulation_type=sim_type, table=simulation.databases.SCAN_TABLE)
    params = sql.get_max(db_path=scan_path, table=simulation.databases.BEST_PARAM_TABLE, group_cols=[],
                         select_cols=select_cols, max_col=keys.PERFORMANCE, maxmin=True)[0]

    # Run final simulations and save performance for each task
    outcomes = np.zeros((constants.NR_SEEDS, sum(TRIAL_NRS)))
    for s in range(constants.NR_SEEDS):
        print(f'\rFigure 3efg: Simulating seed {s}/{constants.NR_SEEDS}...', end='')
        outcomes[s, :] = simulation.simulate.simulate_params(task_id=TASK_ID, simulation_type=sim_type,
                                                             seed=seeds[s], params=list(params), nr_contexts=2,
                                                             n_trials=TRIAL_NRS)
    print(f'\rFigure 3efg: Simulating {sim_type} finished successfully!')

    conn = sqlite3.connect(save_db_path)
    cur = conn.cursor()
    for s in range(constants.NR_SEEDS):
        values = tuple([seeds[s]] + outcomes[s, :].tolist())
        cur.execute(insert_cmd, values)
    conn.commit()
    cur.close()
    del cur
    conn.close()


def simulate_condition(simulation_type: ids.SimulationType):

    # Initialize database variables
    if simulation_type == NO_OFC_SIM:
        scan_type = ids.ST_FINAL
        seeds = list(range(constants.NR_SEEDS, 2 * constants.NR_SEEDS))

    elif simulation_type == OFC_SIM:
        scan_type = OFC_SCAN
        seeds = list(range(2 * constants.NR_SEEDS, 3 * constants.NR_SEEDS))

    else:
        raise NotImplementedError(simulation_type)
    scan_db_path = simulation.scan.get_db_path(simulation_type=simulation_type, scan_type=scan_type)

    final_simulation(sim_type=simulation_type, scan_path=scan_db_path, seeds=seeds)


def get_data(db_path: Path = SAVE_PATH, tables: List = None, trial_nrs=TRIAL_NRS):
    if tables is None:
        tables = [NO_OFC_SIM, OFC_SIM]

    # Database containing data
    if db_path is None:
        db_path = paths.RESULT_DIR / paths.SIMULATION_SUBDIR / f"{NAME}.db"
    get_cols = [f"{keys.PERFORMANCE}_{t}" for t in range(sum(trial_nrs))]

    # Object to store processed data
    t1_trials = list(range(trial_nrs[0]))
    t2_trials = list(range(trial_nrs[1]))
    t1_xp_t = np.zeros((constants.NR_SEEDS, len(tables)))
    t2_xp_t = np.zeros((constants.NR_SEEDS, len(tables)))
    t1_perf = [np.zeros((constants.NR_SEEDS, trial_nrs[0])) for _ in range(len(tables))]
    t2_perf = [np.zeros((constants.NR_SEEDS, trial_nrs[1])) for _ in range(len(tables))]

    # Get and process data for each condition
    for i in range(len(tables)):

        # Get data
        outcomes = np.array(sql.select(db_path=db_path, table=tables[i], get_cols=get_cols))

        # Process data
        for seed in range(constants.NR_SEEDS):
            # Process first task
            t1_perf[i][seed, :] = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, :trial_nrs[0]])
            t1_xp_t[seed, i] = ofcsst.utils.process.get_expert_t(performances=t1_perf[i][seed, :])

            # Process second task
            t2_perf[i][seed, :] = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, trial_nrs[0]:])
            t2_xp_t[seed, i] = ofcsst.utils.process.get_expert_t(performances=t2_perf[i][seed, :])

    return t1_trials, t2_trials, t1_xp_t, t2_xp_t, t1_perf, t2_perf


def plot_performance_bars(save: bool = True, db_path: Path = SAVE_PATH, panel='x', plot_name: str = 'perf_bar',
                          plot_dir: Path = PLOT_DIR, colors: List = None, labels: List[str] = None,
                          verbose: bool = False) -> None:

    # Initializations
    n_sims = 2
    if colors is None:
        colors = [COLOR_NO_OFC, COLOR_WT]

    # Get data
    _, _, _, _, t1_perf, t2_perf = get_data(db_path=db_path)
    learning_perf = [None, None]
    reversal_perf = [None, None]
    t = int(TRIAL_NRS[0] - constants.PERF_MAV_N / 2.) - 1
    for i in range(n_sims):
        learning_perf[i] = t1_perf[i][:, t]
        reversal_perf[i] = t2_perf[i][:, t]

    # Init plot
    helper.set_style()
    fig, axs = plt.subplots(1, 2, figsize=(0.5 * helper.PANEL_WIDTH, helper.PANEL_HEIGHT))

    # Plot performance bars
    xs = [1, 0]
    for i in range(n_sims):
        y_means = [np.mean(learning_perf[i]), np.mean(reversal_perf[i])]
        y_std_errs = [np.std(learning_perf[i]), np.std(reversal_perf[i])]
        for j in range(2):
            axs[j].bar(xs[i], y_means[j], yerr=y_std_errs[j], width=0.9, color=colors[i])

    # Compute significant difference statistics
    tt1 = stats.ttest_rel(learning_perf[1], learning_perf[0])
    tt2 = stats.ttest_rel(reversal_perf[1], reversal_perf[0])
    tt3 = stats.ttest_rel(learning_perf[0], reversal_perf[0])
    tt4 = stats.ttest_rel(learning_perf[1], reversal_perf[1])
    pvs = [tt1[1], tt2[1], tt3[1]]
    n_test = len(pvs)
    pvs = [p * n_test for p in pvs]
    if verbose:
        print(f'\nTesting performance differences with or without OFC and during learning or reversal')
        print(f'learning: ofc vs. no ofc has p-value {tt1[1]:.1e}')
        print(f'reversal: ofc vs. no ofc has p-value {tt2[1]:.1e}')
        print(f'ofc: learning vs. reversal has p-value {tt3[1]:.1e}')
        print(f'no ofc: learning vs. reversal has p-value {tt4[1]:.1e}')

    # Plot significance stars / n.s
    y_max = 150
    xs = [[0, 1], [3, 4], [1, 4]]
    y1 = y_max - 48
    y2 = y_max - 38
    ys = [y1, y1, y2]
    axs[0].plot([2, 2], [0, y1], clip_on=False, **helper.STYLE_RULE_SWITCH)
    for i in range(3):
        axs[0].plot(xs[i], [ys[i], ys[i]], clip_on=False, **helper.STYLE_SIGNIFICANT)
        if pvs[i] < constants.SIGNIFICANCE_THRESHOLD:
            axs[0].text(sum(xs[i]) / 2., ys[i], s=helper.get_significance(pvs[i]), **helper.FONT_SIGNIFICANT)
        else:
            axs[0].text(sum(xs[i]) / 2., ys[i] + 2, **helper.FONT_NON_SIGNIFICANT)

    if labels is None:
        labels = ['Intact\nlOFC', 'Silenced\nlOFC']
    colors = [colors[1], colors[0]]
    x_circle = [-0.5, 0.5]
    circ = [None, None]
    for i in range(2):
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].set_xlim([-1, 2])
        circ[i] = patches.Ellipse(xy=(x_circle[i], 145.), width=3, height=40, facecolor=colors[i], clip_on=False)
        axs[i].add_patch(circ[i])
        axs[i].text(x_circle[i], 143, labels[i], ha='center', va='center')
        axs[i].set_ylim([0, y_max])
    axs[1].set_yticks([])
    axs[0].set_yticks([0., 100.])
    axs[0].tick_params(axis='y', which='major', pad=0.5)
    axs[0].spines[['left']].set_bounds([0, 100])
    axs[1].spines[['left']].set_visible(False)
    axs[0].set_ylabel("Performance (Trial #600)", labelpad=1, loc='bottom')
    axs[0].set_xticks([0.5], ["Learning"])
    axs[1].set_xticks([0.5], ["Reversal"])
    axs[1].set_zorder(-1)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.1, wspace=0.)
    fig.subplots_adjust(top=0.85, bottom=0.12)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=plot_dir, plot_name=plot_name)


def panel_e_performance_bars(save: bool = True) -> None:
    plot_performance_bars(save=save, panel='e', plot_name="e_performance_bars")


def plot_cumulative_distribution(save: bool = True, db_path: Path = SAVE_PATH, panel='x', plot_name: str = 'cum',
                                 phase_cosmetics: bool = True, plot_dir: Path = PLOT_DIR, colors: List = None,
                                 labels: List = None) -> None:

    # Basic initializations
    simulation_types = [OFC_SIM, NO_OFC_SIM]
    if colors is None:
        colors = [COLOR_WT, COLOR_NO_OFC]
    if phase_cosmetics:
        k_max = 117.3
    else:
        k_max = 100
    rect_lim = [100, k_max]

    # Database and columns containing data
    get_cols = [f"{keys.PERFORMANCE}_{t}" for t in range(sum(TRIAL_NRS))]

    # Plot expert time pyramid
    helper.set_style()
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(0.8 * helper.PANEL_WIDTH, helper.PANEL_HEIGHT),
                            height_ratios=TRIAL_NRS, width_ratios=[1, 1, 0.2])
    for j in range(len(TRIAL_NRS)):
        n_trials = TRIAL_NRS[j]
        t2_xp_t = [None, None]
        bin_width = TRIAL_NRS[1] / 32.
        n_bins = int(n_trials / bin_width)
        for i in range(len(simulation_types)):
            all_outcomes = np.array(sql.select(db_path=db_path, table=simulation_types[i], get_cols=get_cols))
            if j == 0:
                outcomes = all_outcomes[:, :TRIAL_NRS[0]]
            else:
                outcomes = all_outcomes[:, TRIAL_NRS[0]:]
            nr_seeds = outcomes.shape[0]
            t2_xp_t[i] = np.zeros((nr_seeds,))
            for seed in range(nr_seeds):
                performances = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, :])
                t2_xp_t[i][seed] = ofcsst.utils.process.get_expert_t(performances=performances)

            # Extract cumulative distribution
            histo = np.histogram(t2_xp_t[i], bins=n_bins, range=(0, n_trials))
            xs = histo[1][1:] - 0.5 * n_trials / n_bins
            cumulative_dist = 100 * np.cumsum(histo[0]) / nr_seeds

            # Plot pyramid
            axs[j, i].barh(xs, cumulative_dist, color=colors[i], height=bin_width, linewidth=0.5, edgecolor="k",
                           clip_on=False)

            # Set axes
            axs[j, i].set_xlim([0, 100])
            axs[j, i].set_ylim([0, n_trials])
            axs[j, i].spines[['right', 'top']].set_visible(False)
            axs[j, i].invert_yaxis()

        # More axes cosmetics
        axs[j, 0].invert_xaxis()
        axs[j, 1].set_yticks([])
        axs[0, j].set_xticks([])
        axs[j, 2].axis('off')
        axs[j, 2].invert_yaxis()

        # Plot cosmetics
        if phase_cosmetics:
            rect = patches.Rectangle(xy=(0, constants.NR_TRIALS), width=rect_lim[j],
                                     height=TRIAL_NRS[1] - constants.NR_TRIALS, linewidth=0, facecolor=(0., 0., 0.),
                                     alpha=0.2, clip_on=False)
            axs[1, j].add_patch(rect)
    axs[0, 0].set_yticks([1, constants.NR_TRIALS])
    axs[1, 0].set_yticks([1, constants.NR_TRIALS, TRIAL_NRS[1]])
    fig.supylabel("Trials (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.05, y=0.55)
    axs[1, 1].plot([-100, k_max], [-140, -140], clip_on=False, **helper.STYLE_RULE_SWITCH)
    axs[1, 0].text(50, -50, clip_on=False, **helper.FONT_RULE_SWITCH)
    axs[1, 1].plot([0, 0], [-300, 2], 'k', clip_on=False, linewidth=helper.AXIS_WIDTH)

    # Learning phase cosmetics
    if phase_cosmetics:
        norm = Normalize(vmin=0, vmax=1)
        cbar_style = {'fraction': 1., 'ticks': [], 'aspect': 4, 'anchor': (0.5, 1)}
        cb = plt.colorbar(ScalarMappable(norm=norm, cmap=helper.get_performance_cmap(phase='learning').reversed()),
                          ax=axs[1, 2], shrink=0.5, **cbar_style)
        cb.ax.zorder = -1
        cb.outline.set_visible(False)
        cb = plt.colorbar(ScalarMappable(norm=norm, cmap=helper.get_performance_cmap(phase='reversal').reversed()),
                          ax=axs[0, 2], **cbar_style)
        cb.ax.zorder = -1
        cb.outline.set_visible(False)
        font = {'x': -9.8, 'fontsize': 4, 'color': 'white', 'ha': 'center', 'va': 'center'}
        axs[0, 2].text(y=0.14, s='LN', **font)
        axs[0, 2].text(y=0.9, s='LE', **font)
        axs[1, 2].text(y=0.07, s='RN', **font)
        axs[1, 2].text(y=0.45, s='RE', **font)

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0., hspace=0.3)
    fig.subplots_adjust(bottom=0.2)
    if not phase_cosmetics:
        fig.subplots_adjust(right=0.99)
    helper.set_panel_label(label=panel, fig=fig)

    # Last labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Fraction experts [%]")
    if labels is None:
        labels = list(SIM_NAMES.values())
    axs[0, 0].text(50, 30, labels[0], ha='center', clip_on=False)
    axs[0, 1].text(50, 30, labels[1], ha='center', clip_on=False)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=plot_dir, plot_name=plot_name)


def panel_f_cumulative_distribution(save: bool = True) -> None:
    plot_cumulative_distribution(save=save, panel='f', plot_name='f_expertise_cumulative_distribution')


def plot_expert_time(save: bool = True, db_path: Path = SAVE_PATH, panel='x', plot_name: str = 'exp_time',
                     panel_width: float = 0.9 * helper.PANEL_WIDTH, plot_dir: Path = PLOT_DIR) -> None:

    # Basic initializations
    simulation_types = [NO_OFC_SIM, OFC_SIM]
    colors = [COLOR_NO_OFC, COLOR_WT]

    # Get data
    t1_trials, t2_trials, t1_xp_t, t2_xp_t, t1_perf, t2_perf = get_data(db_path=db_path)
    t1_perf_mean = np.zeros((2, len(t1_trials)))
    t2_perf_mean = np.zeros((2, len(t2_trials)))
    for i in range(len(simulation_types)):
        t1_perf_mean[i, :] = np.mean(t1_perf[i], axis=0)
        t2_perf_mean[i, :] = np.mean(t2_perf[i], axis=0)

    # Init plot
    helper.set_style()
    height_scale = 1.
    miny0, max_y0 = 0, max(np.max(t1_perf), np.max(t2_perf))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(panel_width, height_scale * helper.PANEL_HEIGHT),
                            height_ratios=[4 * height_scale, 1], width_ratios=[1, 2])

    # Plot all performance traces
    for seed in range(constants.NR_SEEDS):
        for i in range(len(simulation_types)):
            axs[0, 0].plot(t1_trials, t1_perf[i][seed, :], color=colors[i], alpha=0.05)
            axs[0, 1].plot(t2_trials, t2_perf[i][seed, :], color=colors[i], alpha=0.05)

    # Plot mean performance traces
    for i in [1, 0]:
        axs[0, 0].plot(t1_trials, t1_perf_mean[i, :], color=colors[i], label=SIM_NAMES[simulation_types[i]],
                       linewidth=1.5 * helper.LINE_WIDTH)
        axs[0, 1].plot(t2_trials, t2_perf_mean[i, :], color=colors[i], label=SIM_NAMES[simulation_types[i]],
                       linewidth=1.5 * helper.LINE_WIDTH)

    # Plot some cosmetics
    rect_style = dict(height=1., linewidth=0, alpha=0.4)
    rect = patches.Rectangle(xy=(t1_trials[0], 0.5), width=t1_trials[-1], facecolor=COLOR_WT, **rect_style)
    axs[1, 0].add_patch(rect)
    rect = patches.Rectangle(xy=(t1_trials[0], 1.5), width=t1_trials[-1], facecolor=COLOR_NO_OFC, **rect_style)
    axs[1, 0].add_patch(rect)
    rect = patches.Rectangle(xy=(t2_trials[0], 0.5), width=t2_trials[-1], facecolor=COLOR_WT, **rect_style)
    axs[1, 1].add_patch(rect)
    rect = patches.Rectangle(xy=(t2_trials[0], 1.5), width=t2_trials[-1], facecolor=COLOR_NO_OFC, **rect_style)
    axs[1, 1].add_patch(rect)
    axs[1, 1].plot([t2_trials[0], t2_trials[0]], [-7.5 * height_scale, 2.5], clip_on=False, zorder=10,
                   **helper.STYLE_RULE_SWITCH)
    axs[0, 0].text(constants.NR_TRIALS - 45 * helper.PANEL_WIDTH / panel_width, 20, rotation=90,
                   **helper.FONT_RULE_SWITCH)
    for i in range(2):
        axs[0, i].plot([0, TRIAL_NRS[i] - 1], [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
                       label='Expert threshold', zorder=0, **helper.STYLE_EXPERT_PERF)

    # Plot box plots of time to reach expertise
    bp_props = {'vert': False,
                'widths': 0.5,
                'boxprops': dict(linewidth=helper.AXIS_WIDTH, color='black'),
                'capprops': dict(linewidth=helper.AXIS_WIDTH),
                'whiskerprops': dict(linewidth=helper.AXIS_WIDTH),
                'flierprops': dict(marker='o', markersize=helper.MARKER_SIZE, linewidth=helper.AXIS_WIDTH),
                'medianprops': dict(color='k', solid_capstyle='butt', linewidth=helper.AXIS_WIDTH)}
    pos = [[2], [1]]
    for i in range(len(simulation_types)):
        axs[1, 0].boxplot(t1_xp_t[~np.isnan(t1_xp_t[:, i]), i], positions=pos[i], **bp_props)
        n_reversal_experts = len(t2_xp_t[~np.isnan(t2_xp_t[:, i]), i])
        if n_reversal_experts > constants.NR_SEEDS / 2:
            axs[1, 1].boxplot(t2_xp_t[~np.isnan(t2_xp_t[:, i]), i], positions=pos[i], **bp_props)
        else:
            axs[1, 1].scatter(t2_xp_t[~np.isnan(t2_xp_t[:, i]), i], pos[i] * n_reversal_experts, s=helper.MARKER_SIZE,
                              ec='k', linewidth=helper.AXIS_WIDTH, marker='o')

    # Legend and labels
    axs[0, 0].set_ylabel("Performance")
    axs[1, 0].set_ylabel('Trials to\nexpertise', rotation=0, y=0.1, labelpad=15)
    fig.supxlabel("Trials (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.55, y=0.03)
    axs[0, 1].legend(edgecolor=(0.2, 0.2, 0.2), loc='lower right', frameon=False, borderpad=0, labelspacing=0.3)

    # Set axes
    axs[0, 0].set_xlim([t1_trials[0], t1_trials[-1]])
    axs[0, 1].set_xlim([t2_trials[0], t2_trials[-1]])
    axs[1, 0].set_xlim([t1_trials[0], t1_trials[-1]])
    axs[1, 1].set_xlim([t2_trials[0], t2_trials[-1]])
    axs[0, 0].set_ylim([miny0, max_y0])
    axs[0, 1].set_ylim([miny0, max_y0])
    axs[1, 0].set_ylim([0.5, 2.5])
    axs[1, 1].set_ylim([0.5, 2.5])
    axs[1, 0].invert_yaxis()
    axs[1, 1].invert_yaxis()
    axs[0, 0].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[0, 1].spines[['bottom', 'left', 'right', 'top']].set_visible(False)
    axs[1, 0].spines[['right', 'top']].set_visible(False)
    axs[1, 1].spines[['left', 'right', 'top']].set_visible(False)
    axs[0, 0].spines['left'].set_bounds(miny0, 100)
    axs[0, 0].set_xticks([])
    axs[0, 1].set_xticks([])
    axs[1, 0].set_xticks([0, 100, 200, 300, 400, 500], ['', '100', '', '', '400', ''])
    axs[1, 1].set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                         ['', '100', '', '', '400', '', '', '700', '', '', '1000', '', ''])
    axs[0, 0].set_yticks([0., 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.],
                         ['', '10', '', '30', '', '50', '', '70', '', '90', ''])

    axs[0, 1].set_yticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].set_yticks([])

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0)
    fig.subplots_adjust(bottom=0.2)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=plot_dir, plot_name=plot_name, plot_dpi=500)


def panel_g_expert_time(save: bool = True) -> None:
    plot_expert_time(save=save, panel='g', plot_name='g_expert_time')


def scan():
    simulation.scan.find_best_params(simulation_type=NO_OFC_SIM, task_id=TASK_ID, switch_task=True,
                                     scan_type=ids.ST_FINAL)
    simulation.scan.find_best_params(simulation_type=OFC_SIM, task_id=TASK_ID, switch_task=True, scan_type=OFC_SCAN)


def run():
    print('Running simulations for Figure 3efg')
    simulate_condition(simulation_type=OFC_SIM)
    simulate_condition(simulation_type=NO_OFC_SIM)


def plot(save: bool = True):
    panel_e_performance_bars(save=save)
    panel_f_cumulative_distribution(save=save)
    panel_g_expert_time(save=save)
