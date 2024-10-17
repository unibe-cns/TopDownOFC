import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
from pathlib import Path

import ofcsst.utils.process
from ofcsst.simulation import simulate, scan, databases
from ofcsst.utils import ids, keys, paths, sql, constants
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import OFC_SCAN
import scipy.stats as stats

NAME = "double_reversal"
PLOT_DIR = paths.MAIN_FIG_DIR / "fig4"
NR_TRIALS = (constants.NR_TRIALS, constants.NR_TRIALS, constants.NR_TRIALS)
TABLES = [keys.PERFORMANCE, f"{keys.AGENT_GAIN}_T1", f"{keys.AGENT_GAIN}_T2", f"{keys.AGENT_POLICY_WEIGHT}_T1",
          f"{keys.AGENT_POLICY_WEIGHT}_T2"]
NR_CONTEXTS = 3
SINGLE_C_SIM = ids.SIM_CONVEX_OFC
COLOR_SINGLE = helper.COLOR_DEFAULT


def get_path(simulation_type: ids.SimulationType) -> Path:
    return paths.RESULT_DIR / paths.SIMULATION_SUBDIR / f'fig_4ac_{NAME}_{simulation_type}.db'


def init_sim(simulation_type: ids.SimulationType, slow_pg: bool):
    if simulation_type in [ids.SIM_CONVEX_OFC, ids.SIM_FAKE_TD_SWITCH, ids.SIM_FAKE_XAP_SWITCH,
                           ids.SIM_OFC_XAP_SWITCH]:
        scan_sim_type = ids.SIM_CONVEX_OFC
        scan_type = OFC_SCAN
    elif simulation_type in [ids.SIM_CONVEX_GAIN]:
        scan_sim_type = ids.SIM_CONVEX_GAIN
        scan_type = ids.ST_FINAL
    else:
        raise NotImplementedError

    if simulation_type == ids.SIM_CONVEX_OFC:
        sim_name = 'single-context'
        seeds = list(range(2 * constants.NR_SEEDS, 3 * constants.NR_SEEDS))
    elif simulation_type in [ids.SIM_OFC_XAP_SWITCH]:
        sim_name = 'ofc gated xap switch'
        seeds = list(range(3 * constants.NR_SEEDS, 4 * constants.NR_SEEDS))
    elif simulation_type in [ids.SIM_FAKE_XAP_SWITCH]:
        sim_name = 'fake xap switch'
        seeds = list(range(4 * constants.NR_SEEDS, 5 * constants.NR_SEEDS))
    elif simulation_type == ids.SIM_FAKE_TD_SWITCH:
        sim_name = 'fake td switch'
        seeds = list(range(5 * constants.NR_SEEDS, 6 * constants.NR_SEEDS))
    elif simulation_type == ids.SIM_CONVEX_GAIN:
        sim_name = 'no ofc'
        seeds = list(range(6 * constants.NR_SEEDS, 7 * constants.NR_SEEDS))
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


def get_cols(n_trials: int, with_seed: bool = True) -> dict:
    if with_seed:
        return {TABLES[0]: [str(keys.SEED)] + [f"{keys.PERFORMANCE}_{t}" for t in range(n_trials)],
                TABLES[1]: [str(keys.SEED)] + [f"{keys.AGENT_GAIN}_{t}" for t in range(n_trials)],
                TABLES[2]: [str(keys.SEED)] + [f"{keys.AGENT_GAIN}_{t}" for t in range(n_trials)],
                TABLES[3]: [str(keys.SEED)] + [f"{keys.AGENT_POLICY_WEIGHT}_{t}" for t in range(n_trials)],
                TABLES[4]: [str(keys.SEED)] + [f"{keys.AGENT_POLICY_WEIGHT}_{t}" for t in range(n_trials)]}
    else:
        return {TABLES[0]: [f"{keys.PERFORMANCE}_{t}" for t in range(n_trials)],
                TABLES[1]: [f"{keys.AGENT_GAIN}_{t}" for t in range(n_trials)],
                TABLES[2]: [f"{keys.AGENT_GAIN}_{t}" for t in range(n_trials)],
                TABLES[3]: [f"{keys.AGENT_POLICY_WEIGHT}_{t}" for t in range(n_trials)],
                TABLES[4]: [f"{keys.AGENT_POLICY_WEIGHT}_{t}" for t in range(n_trials)]}


def simulate_multi_reversal_analysis(simulation_type: ids.SimulationType, slow_pg: bool = False, nr_reversals: int = 2,
                                     save_db_path: Path = None):
    # Initializations
    nr_contexts = nr_reversals + 1
    scan_sim_type, scan_type, sim_name, seeds, params = init_sim(simulation_type=simulation_type, slow_pg=slow_pg)
    col_keys = get_cols(n_trials=constants.NR_TRIALS * nr_contexts, with_seed=True)
    if save_db_path is None:
        save_db_path = get_path(simulation_type=simulation_type)
    conn = sql.connect(db_path=save_db_path)
    cur = conn.cursor()
    insert_cmd = {}
    for table in TABLES:
        sql.drop_table(conn=conn, table_name=table, verbose=False)
        sql.make_table(conn=conn, table_name=table, col_names=col_keys[table], verbose=False)
        insert_cmd[table] = sql.get_insert_cmd(table=table, col_keys=col_keys[table])

    # Run simulations for each seed
    for s in range(constants.NR_SEEDS):
        print(f"\rSimulating {sim_name} {100 * s / constants.NR_SEEDS:0.1f}% completed", end="")

        # Simulate seed
        rewards, (gain_t1, gain_t2, policy_weights_t1, policy_weights_t2, _) = simulate.run_simulation(
            task_id=constants.TASK_ID, simulation_type=simulation_type, params=params, seed=seeds[s],
            number_contexts=nr_contexts, log_type=ids.LT_AGENT
        )

        # Append outcomes to database
        values = tuple([seeds[s]] + rewards.tolist())
        cur.execute(insert_cmd[TABLES[0]], values)
        values = tuple([seeds[s]] + gain_t1.tolist())
        cur.execute(insert_cmd[TABLES[1]], values)
        values = tuple([seeds[s]] + gain_t2.tolist())
        cur.execute(insert_cmd[TABLES[2]], values)
        values = tuple([seeds[s]] + policy_weights_t1.tolist())
        cur.execute(insert_cmd[TABLES[3]], values)
        values = tuple([seeds[s]] + policy_weights_t2.tolist())
        cur.execute(insert_cmd[TABLES[4]], values)

    # Close database
    conn.commit()
    cur.close()
    del cur
    conn.close()

    print(f"\rSimulating {sim_name} is now completed!")


def panel_c_traces(save: bool = True) -> None:

    # Initializations
    simulation_type = SINGLE_C_SIM
    n_trials = constants.NR_TRIALS
    db_path = get_path(simulation_type=simulation_type)
    cols = get_cols(n_trials=sum(NR_TRIALS), with_seed=False)
    task_trials = list(range(3 * n_trials))
    context_trials = [list(range(i * n_trials, (i + 1) * n_trials)) for i in range(NR_CONTEXTS)]
    temp_perfs = np.zeros((NR_CONTEXTS, constants.NR_SEEDS, n_trials))
    t_ranges = [np.array(list(range(r * n_trials, (r + 1) * n_trials))) for r in
                range(NR_CONTEXTS)]
    rows = [None, 1, 1, 2, 2]
    color = 'k'
    fill_color = [(247. / 255., 231. / 255., 153 / 255.), (144. / 255., 182. / 255., 165. / 255.)]

    linestyle_original = 'dashed'
    linestyle_reversed = 'dotted'
    linestyles = [None, linestyle_original, linestyle_reversed, linestyle_original, linestyle_reversed]

    # Process performances
    task_switches = list(range(NR_CONTEXTS))
    outcomes = np.array(sql.select(db_path=db_path, table=TABLES[0], get_cols=cols[TABLES[0]]))
    for seed in range(constants.NR_SEEDS):
        for t in task_switches:
            temp_perfs[t, seed, :] = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, t_ranges[t]])
    seed_performances = np.concatenate([temp_perfs[t, :, :] for t in task_switches], axis=1)

    # Init plot
    n_rows = 3
    helper.set_style()
    with rc_context({'mathtext.fontset': 'cm'}):
        fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(helper.PANEL_WIDTH, 1.35 * helper.PANEL_HEIGHT),
                                height_ratios=(0.7, 1, 1))
    axs[0].plot([0, 3 * n_trials], [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
                **helper.STYLE_EXPERT_PERF)

    # Plot performance traces
    for c in range(NR_CONTEXTS):
        ts = context_trials[c]
        mv = np.mean(seed_performances[:, ts], axis=0)
        stderr = np.std(seed_performances[:, ts], axis=0)
        axs[0].fill_between(ts, mv - stderr, mv + stderr, linewidth=0, color=(0.7, 0.7, 0.7))
        axs[0].plot(ts, mv, color=color)

    # Plot reversal line cosmetics
    ymin = np.min(temp_perfs)
    ymax = np.max(temp_perfs)
    y_lims = [[ymin, ymax], [-0.7, 12], [-0.1, 1.1]]
    axs[0].set_zorder(100)
    axs[2].set_zorder(100)
    for row in range(n_rows):
        axs[row].patch.set_alpha(0.)
        axs[row].set_xlim([0, 3 * n_trials])
        axs[row].set_ylim(y_lims[row])
    axs[0].text(n_trials - 38, 30, rotation=90, **helper.FONT_RULE_SWITCH)
    axs[0].text(2 * n_trials - 38, 30, rotation=90, **helper.FONT_RULE_SWITCH)
    for x in [n_trials - 0.5, 2 * n_trials - 0.5]:
        axs[2].plot([x, x], [y_lims[2][0], y_lims[2][1] * 3.3], clip_on=False, zorder=2, **helper.STYLE_RULE_SWITCH)

    # Plot traces of SNR and pi weight
    labels: list = [None for _ in range(len(TABLES))]
    labels[-2] = 'Original rule'
    labels[-1] = 'Reversed rule'
    for table_id in range(1, len(TABLES)):
        values = np.array(sql.select(db_path=db_path, table=TABLES[table_id], get_cols=cols[TABLES[table_id]]))
        mv = np.mean(values, axis=0)[task_trials]
        stderr = np.std(values, axis=0)[task_trials]
        axs[rows[table_id]].plot(task_trials, mv, linestyle=linestyles[table_id], color=color,
                                 label=labels[table_id])
        axs[rows[table_id]].fill_between(task_trials, mv - stderr, mv + stderr, color=fill_color[rows[table_id] - 1],
                                         linewidth=0)

    axs[0].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[1].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[2].spines[['right', 'top']].set_visible(False)
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([100 + 200 * i for i in range(9)], ['100', '300', '500'] * 3)
    axs[0].spines['left'].set_bounds(0, 100)
    axs[0].set_yticks([0, 100])
    axs[0].set_ylabel("Performance", va='baseline')
    axs[1].spines['left'].set_bounds(0, 10)
    axs[1].set_yticks([0, 10])
    axs[1].set_ylabel(r"Rel. gain $\Gamma_\mathrm{go}^\mathrm{ap}$", va='baseline')
    axs[2].spines['left'].set_bounds(y_lims[2][0], 1)
    axs[2].set_yticks([0, 1])
    axs[2].set_ylabel(r"$\pi$-weight", va='baseline')
    axs[2].set_xlabel("Trials (after start/reversal)")
    axs[0].set_title("  Learning          Reversal 1        Reversal 2", fontsize=helper.FONT_SIZE)
    fig.align_ylabels(axs)
    for ax in axs:
        ax.yaxis.set_label_coords(-0.15, 0.5)

    # Plot cosmetics like legends and reversal lines
    leg = axs[2].legend(loc="lower left", framealpha=1, ncol=2, handlelength=2.4, bbox_to_anchor=(0., 0.82, 1., 0.5))
    axs[2].add_artist(leg)
    lines = [plt.Line2D([], [], linestyle=ls, color='k') for ls in [linestyle_original, linestyle_reversed]]
    with rc_context({'mathtext.fontset': 'cm'}):
        labels = [r'$\Gamma_\mathrm{s_1}^\mathrm{ap}$', r'$\Gamma_\mathrm{s_2}^\mathrm{ap}$']
        legend_arg = {'loc': 'lower right', 'frameon': False, 'handlelength': 1.7}
        axs[1].legend(handles=lines, labels=labels, bbox_to_anchor=(0.5, 0.1, 0.5, 0.5), **legend_arg)
        labels = [r'$w_\mathrm{s_1}^\mathrm{PG}$', r'$w_\mathrm{s_2}^\mathrm{PG}$']
        axs[2].legend(handles=lines, labels=labels, bbox_to_anchor=(0.5, -0.05, 0.5, 0.5), **legend_arg)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.1, wspace=0)
    fig.subplots_adjust(bottom=0.13, left=0.2)
    helper.set_panel_label(label="c", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="c_traces", plot_dpi=400)


def panel_d_expert_times(save: bool = True, verbose: bool = False) -> None:

    # Init plot
    helper.set_style()
    fig_width, fig_height = 1. * helper.PANEL_WIDTH, 0.6 * helper.PANEL_HEIGHT
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height))
    xp_id, sim_id = 0, 1

    # Get trials when expertise was reached by model in simulations
    db_path = get_path(simulation_type=SINGLE_C_SIM)
    cols = [f"{keys.PERFORMANCE}_{t}" for t in range(sum(NR_TRIALS))]
    xp_t = {}
    for ts in range(NR_CONTEXTS + 1):
        xp_t[ts] = np.zeros(constants.NR_SEEDS)
    t_ranges = [np.array(list(range(r * constants.NR_TRIALS, (r + 1) * constants.NR_TRIALS))) for r in
                range(NR_CONTEXTS)]
    outcomes = np.array(sql.select(db_path=db_path, table=TABLES[0], get_cols=cols))
    perfs = np.zeros((NR_CONTEXTS, constants.NR_SEEDS, constants.NR_TRIALS))
    for seed in range(constants.NR_SEEDS):
        for t in range(NR_CONTEXTS):
            perfs[t, seed, :] = ofcsst.utils.process.get_performance(outcomes[seed, t_ranges[t]])
            xp_t[t][seed] = ofcsst.utils.process.get_expert_t(perfs[t, seed, :])

    # Plot boxes of trials when expertise was reached by model in simulation
    bp_props = {'vert': False,
                'widths': 0.5,
                'boxprops': dict(linewidth=helper.LINE_WIDTH, color='black'),
                'capprops': dict(linewidth=helper.LINE_WIDTH),
                'whiskerprops': dict(linewidth=helper.LINE_WIDTH),
                'flierprops': dict(marker='o', markersize=helper.MARKER_SIZE, linewidth=helper.LINE_WIDTH),
                'medianprops': dict(color='k', solid_capstyle='butt')}
    bp_props['boxprops']['facecolor'] = COLOR_SINGLE
    for ts in range(NR_CONTEXTS):
        xp_t[ts] = xp_t[ts][~np.isnan(xp_t[ts])]
        axs[sim_id].boxplot(np.transpose(xp_t[ts]), positions=[ts + 1], patch_artist=True, **bp_props)

    # Plot significance stars
    idxs = [[0, 1], [0, 2], [1, 2]]
    n_test = len(idxs)
    pvs = [stats.ttest_ind(xp_t[i[0]], xp_t[i[1]])[1] * n_test for i in idxs]
    if verbose:
        print(f'For model Learning vs. Reversal 1: p-value is {pvs[0]:.1e}')
        print(f'For model Learning vs. Reversal 2: p-value is {pvs[1]:.1e}')
        print(f'For model Reversal 1 vs. Reversal 2: p-value is {pvs[2]:.1e}')
    ref_x = constants.NR_TRIALS - 160
    xs = [ref_x + 10, ref_x + 120, ref_x + 65, ref_x + 10]
    ys = [[1, 2], [1, 3], [2, 3]]
    significance_style = dict(color="k", linewidth=0.8 * helper.LINE_WIDTH)
    for i in range(n_test):
        axs[sim_id].plot([xs[i], xs[i]], ys[i], **significance_style)
        if pvs[i] < constants.SIGNIFICANCE_THRESHOLD:
            axs[sim_id].text(xs[i], sum(ys[i]) / 2., s=helper.get_significance(pvs[i]), rotation=270,
                             **helper.FONT_SIGNIFICANT)
        else:
            axs[sim_id].text(xs[i], sum(ys[i]) / 2., rotation=270, **helper.FONT_NON_SIGNIFICANT)

    # Axis cosmetics
    xlim = [[], []]
    xlim[xp_id] = [0, 9]
    xlim[sim_id] = [0, constants.NR_TRIALS]
    for r in range(2):
        axs[r].invert_yaxis()
        axs[r].spines[['right', 'top']].set_visible(False)
        axs[r].tick_params(axis='both', which='major', pad=1)
        axs[r].set_xlim(xlim[r])
    axs[sim_id].set_xticks([0, 100, 200, 300, 400, 500, 600], ['', '100', '', '300', '', '500', ''])
    axs[xp_id].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['', '1', '', '3', '', '5', '', '7', '', '9'])
    axs[sim_id].set_xlabel("Trials to expertise")
    axs[xp_id].set_xlabel("Sessions to expertise")
    axs[1].set_yticklabels([])
    axs[0].set_ylim([-0.5, 2.5])
    axs[0].set_yticks([2, 1, 0])
    axs[0].set_yticklabels(["Learning", "Reversal 1", "Reversal 2"])

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0.1)
    fig.subplots_adjust(top=0.8)
    helper.set_panel_label(label="d", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="d_expertise")


def run():
    print('Simulating for Figure 4cd:')
    simulate_multi_reversal_analysis(simulation_type=SINGLE_C_SIM)


def plot(save: bool = True):
    panel_c_traces(save=save)
    panel_d_expert_times(save=save)
