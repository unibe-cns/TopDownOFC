import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
from pathlib import Path

import ofcsst.utils.process
from ofcsst.simulation import simulate, scan, databases
from ofcsst.utils import ids, keys, paths, sql, constants
from ofcsst.figures import helper
from scipy import stats

NAME = "double_reversal"
PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs7"
TABLES = [keys.PERFORMANCE, f"{keys.AGENT_POLICY_WEIGHT}_T1", f"{keys.AGENT_POLICY_WEIGHT}_T2"]
NR_CONTEXTS = 3
SIM_TYPE = ids.SIM_CONVEX_NO_GAIN
COLOR_SINGLE = 'silver'
N_TRIALS = 200


def get_path() -> Path:
    return paths.SUPPLEMENTARY_SIMULATION_DIR / 'fig_s7_no_gain.pickle'


def init_sim(simulation_type: ids.SimulationType, slow_pg: bool):
    if simulation_type in [ids.SIM_CONVEX_NO_GAIN]:
        scan_sim_type = ids.SIM_CONVEX_NO_GAIN
        scan_type = ids.ST_NO_DISTRACTOR
    else:
        raise NotImplementedError

    if simulation_type == ids.SIM_CONVEX_NO_GAIN:
        sim_name = 'convex policy with no gain'
        seeds = list(range(7 * constants.NR_SEEDS, 8 * constants.NR_SEEDS))
    else:
        raise NotImplementedError

    # Get the best parameters for each type of simulation
    scan_db_path = scan.get_db_path(scan_type=scan_type, simulation_type=scan_sim_type, task_id=constants.TASK_ID,
                                    non_stationary=True)
    group_cols = [keys.NR_DISTRACTOR]
    select_cols = databases.get_unique_cols(simulation_type=scan_sim_type, table=databases.SCAN_TABLE)
    params: list = list(sql.get_max(db_path=scan_db_path, table=databases.BEST_PARAM_TABLE, group_cols=group_cols,
                                    select_cols=select_cols, max_col=keys.PERFORMANCE, maxmin=True)[0])
    if slow_pg:
        params[6] = params[6] * 0.2

    return scan_sim_type, scan_type, sim_name, seeds, params


def simulate_multi_reversal_analysis():
    simulation_type = SIM_TYPE
    nr_contexts = 3
    scan_sim_type, scan_type, sim_name, seeds, params = init_sim(simulation_type=simulation_type, slow_pg=False)
    results = {'r': [], 'p1': [], 'p2': []}
    # Run simulations for each seed
    for s in range(constants.NR_SEEDS):
        print(f"\rSimulating {sim_name} {100 * s / constants.NR_SEEDS:0.1f}% completed", end="")

        # Simulate seed
        rewards, (policy_weights_t1, policy_weights_t2) = simulate.run_simulation(
            task_id=constants.TASK_ID, simulation_type=simulation_type, params=params, seed=seeds[s],
            number_contexts=nr_contexts, log_type=ids.LT_AGENT, number_trials=(N_TRIALS, N_TRIALS, N_TRIALS)
        )
        results['r'] += [rewards]
        results['p1'] += [policy_weights_t1]
        results['p2'] += [policy_weights_t2]

    with open(get_path(), "wb") as f:
        pickle.dump(results, f)

    print(f"\rSimulating {sim_name} complete!")


def panel_a_pi_weight(save: bool = True) -> None:

    # Initializations
    with open(get_path(), "rb") as f:
        results = pickle.load(f)
    task_trials = list(range(3 * N_TRIALS))
    context_trials = [list(range(i * N_TRIALS, (i + 1) * N_TRIALS)) for i in range(NR_CONTEXTS)]
    temp_perfs = np.zeros((NR_CONTEXTS, constants.NR_SEEDS, N_TRIALS))
    t_ranges = [np.array(list(range(r * N_TRIALS, (r + 1) * N_TRIALS))) for r in
                range(NR_CONTEXTS)]
    mean_color = 'k'
    fill_color = (144. / 255., 182. / 255., 165. / 255.)

    # Process performances
    task_switches = list(range(NR_CONTEXTS))
    outcomes = results['r']
    for seed in range(constants.NR_SEEDS):
        for t in task_switches:
            temp_perfs[t, seed, :] = ofcsst.utils.process.get_performance(outcomes=outcomes[seed][t_ranges[t]])
    seed_performances = np.concatenate([temp_perfs[t, :, :] for t in task_switches], axis=1)

    # Init plot
    n_rows = 2
    helper.set_style()
    with rc_context({'mathtext.fontset': 'cm'}):
        fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))

    # Plot performance traces
    axs[0].plot([0, NR_CONTEXTS * constants.NR_TRIALS], [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
                **helper.STYLE_EXPERT_PERF)
    for c in range(NR_CONTEXTS):
        ts = context_trials[c]
        mv = np.mean(seed_performances[:, ts], axis=0)
        stderr = np.std(seed_performances[:, ts], axis=0)
        axs[0].fill_between(ts, mv - stderr, mv + stderr, linewidth=0, color=(0.7, 0.7, 0.7))
        axs[0].plot(ts, mv, color=mean_color)

    # Plot reversal line cosmetics
    ymin = np.min(temp_perfs)
    ymax = np.max(temp_perfs)
    y_lims = [[ymin, ymax], [-0.1, 13]]
    axs[0].set_zorder(100)
    axs[1].set_zorder(100)
    for row in range(n_rows):
        axs[row].patch.set_alpha(0.)
        axs[row].set_xlim([0, 3 * N_TRIALS])
        axs[row].set_ylim(y_lims[row])
    axs[0].text(N_TRIALS - 13, 30, rotation=90, **helper.FONT_RULE_SWITCH)
    axs[0].text(2 * N_TRIALS - 13, 30, rotation=90, **helper.FONT_RULE_SWITCH)
    for x in [N_TRIALS - 0.5, 2 * N_TRIALS - 0.5]:
        axs[1].plot([x, x], [y_lims[1][0], y_lims[1][1] * 2.42], clip_on=False, zorder=2, **helper.STYLE_RULE_SWITCH)

    # Plot traces of SNR and pi weight
    labels: list = [r'$w_\mathrm{s_1}^\mathrm{PG}$', r'$w_\mathrm{s_2}^\mathrm{PG}$']
    linestyle_original = 'dashed'
    linestyle_reversed = 'dotted'
    linestyles = [linestyle_original, linestyle_reversed]
    ks = ['p1', 'p2']
    for table_id in range(2):
        values = np.stack(results[ks[table_id]])
        mv = np.mean(values, axis=0)[task_trials]
        stderr = np.std(values, axis=0)[task_trials]
        axs[1].plot(task_trials, mv, linestyle=linestyles[table_id], color=mean_color, label=labels[table_id])
        axs[1].fill_between(task_trials, mv - stderr, mv + stderr, color=fill_color, linewidth=0)

    axs[0].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[1].spines[['right', 'top']].set_visible(False)
    axs[0].set_xticks([])

    axs[1].set_xticks([50 * i for i in range(12)], ['', '50', '', '150'] * 3)
    axs[0].spines['left'].set_bounds(0, 100)
    axs[0].set_yticks([0, 100])
    axs[0].set_ylabel("Performance", labelpad=0)
    axs[1].spines['left'].set_bounds(y_lims[1][0], 10)
    axs[1].set_yticks([0, 10])
    axs[1].set_ylabel(r"$\pi$-weight", y=0.4, labelpad=0)
    axs[1].set_xlabel("Trials (after start/reversal)")
    axs[0].set_title("  Learning          Reversal 1        Reversal 2", fontsize=6)
    fig.align_ylabels(axs)

    # Plot cosmetics like legends and reversal lines
    with rc_context({'mathtext.fontset': 'cm'}):
        lines = [plt.Line2D(xdata=[], ydata=[], linestyle=ls, color='k') for ls in linestyles]
        legend_arg = {'loc': 'lower right', 'frameon': False, 'handlelength': 1.7}
        labels = [r'$w_\mathrm{s_1}^\mathrm{PG}$', r'$w_\mathrm{s_2}^\mathrm{PG}$']
        axs[1].legend(handles=lines, labels=labels, bbox_to_anchor=(0.55, -0.05, 0.5, 0.5), **legend_arg)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.1)
    helper.set_panel_label(label="a", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="a_pi_weight", plot_dpi=400)


def panel_b_expert_times(save: bool = True, verbose: bool = False) -> None:

    # Init plot
    helper.set_style()
    fig_width, fig_height = 0.8 * helper.PANEL_WIDTH, 0.5 * helper.PANEL_HEIGHT
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height))

    # Get trials when expertise was reached
    xp_t = {}
    for ts in range(NR_CONTEXTS + 1):
        xp_t[ts] = np.zeros(constants.NR_SEEDS)
    t_ranges = [np.array(list(range(r * N_TRIALS, (r + 1) * N_TRIALS))) for r in range(NR_CONTEXTS)]
    with open(get_path(), "rb") as f:
        results = pickle.load(f)
    outcomes = results['r']
    perfs = np.zeros((NR_CONTEXTS, constants.NR_SEEDS, N_TRIALS))
    for seed in range(constants.NR_SEEDS):
        for t in range(NR_CONTEXTS):
            perfs[t, seed, :] = ofcsst.utils.process.get_performance(outcomes[seed][t_ranges[t]])
            xp_t[t][seed] = ofcsst.utils.process.get_expert_t(perfs[t, seed, :])

    # Plot boxes of trials when expertise was reached by model in simulation
    bp_props = {'vert': False,
                'widths': 0.5,
                'boxprops': dict(linewidth=helper.LINE_WIDTH, color='black', facecolor=COLOR_SINGLE),
                'capprops': dict(linewidth=helper.LINE_WIDTH),
                'whiskerprops': dict(linewidth=helper.LINE_WIDTH),
                'flierprops': dict(marker='o', markersize=helper.MARKER_SIZE, linewidth=helper.LINE_WIDTH),
                'medianprops': dict(color='k', solid_capstyle='butt')}
    for ts in range(NR_CONTEXTS):
        xp_t[ts] = xp_t[ts][~np.isnan(xp_t[ts])]
        axs.boxplot(np.transpose(xp_t[ts]), positions=[ts + 1], patch_artist=True, **bp_props)

    # Plot significance stars
    idxs = [[0, 1], [0, 2], [1, 2]]
    n_test = len(idxs)
    pvs = [stats.ttest_ind(xp_t[i[0]], xp_t[i[1]])[1] * n_test for i in idxs]
    if verbose:
        print('\nTesting difference between expert times for learning, reversal 1 and reversal 2:')
        print(f'Learning vs. Reversal 1: p-value is {pvs[0]:.1e}')
        print(f'Learning vs. Reversal 2: p-value is {pvs[1]:.1e}')
        print(f'Reversal 1 vs. Reversal 2: p-value is {pvs[2]:.1e}')
    ref_x = N_TRIALS - 70
    xs = [ref_x + 20, ref_x + 60, ref_x + 40, ref_x + 20]
    ys = [[1, 2], [1, 3], [2, 3]]
    significance_style = dict(color="k", linewidth=0.8 * helper.LINE_WIDTH)
    for i in range(n_test):
        axs.plot([xs[i], xs[i]], ys[i], **significance_style)
        if pvs[i] < constants.SIGNIFICANCE_THRESHOLD:
            axs.text(xs[i], sum(ys[i]) / 2., s=helper.get_significance(pvs[i]), rotation=270,
                     **helper.FONT_SIGNIFICANT)
        else:
            axs.text(xs[i] + 5, sum(ys[i]) / 2., rotation=270, va='center', **helper.FONT_NON_SIGNIFICANT)

    # Axis cosmetics
    axs.invert_yaxis()
    axs.spines[['right', 'top']].set_visible(False)
    axs.tick_params(axis='both', which='major', pad=1)
    axs.set_xlim([0, N_TRIALS])
    axs.set_xticks([0, 50, 100, 150, 200], ['', '50', '100', '150', '200'])
    axs.set_xlabel("Trials to expertise")
    axs.set_yticklabels(["Learning", "Reversal 1", "Reversal 2"])

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    # fig.subplots_adjust(top=0.8)
    helper.set_panel_label(label="b", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="b_expert_time")


def run():
    print('Running simulations for Figure S7')
    simulate_multi_reversal_analysis()


def plot(save: bool = True):
    panel_a_pi_weight(save=save)
    panel_b_expert_times(save=save)
