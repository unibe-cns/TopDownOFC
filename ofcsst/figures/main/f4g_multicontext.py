import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

import ofcsst.utils.process
from ofcsst.simulation import simulate
from ofcsst.utils import ids, keys, paths, sql, constants
from ofcsst.figures import helper
from ofcsst.figures.main.f4bcd_double_reversal import init_sim, COLOR_SINGLE
from ofcsst.figures.main.f4bcd_double_reversal import get_cols as get_cols_without_ofc

PLOT_DIR = paths.MAIN_FIG_DIR / "fig4"
NR_CONTEXTS = 6
TABLES = [keys.PERFORMANCE, f"{keys.AGENT_GAIN}_T1", f"{keys.AGENT_GAIN}_T2", f"{keys.AGENT_POLICY_WEIGHT}_T1",
          f"{keys.AGENT_POLICY_WEIGHT}_T2", f'{keys.OFC_ACTIVITY}']


def get_path(simulation_type: ids.SimulationType) -> Path:
    return paths.RESULT_DIR / paths.SIMULATION_SUBDIR / f'fig_4e_multi_reversal_{simulation_type}.db'


def get_table(table, context) -> str:
    return f'{table}_{context}'


def get_cols(n_trials: int, with_seed: bool = True) -> dict:
    cols = get_cols_without_ofc(n_trials=n_trials, with_seed=with_seed)
    if with_seed:
        cols[TABLES[5]] = [str(keys.SEED)] + [f"{keys.OFC_ACTIVITY}_{t}" for t in range(n_trials)]
    else:
        cols[TABLES[5]] = [f"{keys.OFC_ACTIVITY}_{t}" for t in range(n_trials)]
    return cols


def run_multi_reversal(simulation_type: ids.SimulationType):

    # Initializations
    scan_sim_type, scan_type, sim_name, seeds, params = init_sim(simulation_type=simulation_type, slow_pg=False)
    save_db_path = get_path(simulation_type=simulation_type)
    col_keys = get_cols(n_trials=constants.NR_TRIALS)
    conn = sql.connect(db_path=save_db_path)
    cur = conn.cursor()
    for table in TABLES:
        for context in range(NR_CONTEXTS):
            table_name = get_table(table=table, context=context)
            sql.drop_table(conn=conn, table_name=table_name, verbose=False)
            sql.make_table(conn=conn, table_name=table_name, col_names=col_keys[table], verbose=False)

    # Run simulations for each seed
    for s in range(constants.NR_SEEDS):
        print(f"\rSimulating {sim_name} {100 * s / constants.NR_SEEDS:0.1f}% completed", end="")

        # Simulate seed
        rewards, (gain_t1, gain_t2, policy_weights_t1, policy_weights_t2, ofc_activity) = simulate.run_simulation(
            task_id=constants.TASK_ID, simulation_type=simulation_type, params=params, seed=seeds[s],
            number_contexts=NR_CONTEXTS, log_type=ids.LT_OFC
        )

        # Append outcomes to database
        record = [rewards.tolist(), gain_t1.tolist(), gain_t2.tolist(), policy_weights_t1.tolist(),
                  policy_weights_t2.tolist(), ofc_activity.tolist()]
        ss = [seeds[s]]
        for t in range(len(TABLES)):
            for context in range(NR_CONTEXTS):
                values = tuple(ss + record[t][context * constants.NR_TRIALS: (context+1) * constants.NR_TRIALS])
                cur.execute(sql.get_insert_cmd(table=get_table(table=TABLES[t], context=context),
                                               col_keys=col_keys[TABLES[t]]), values)

    # Close database
    conn.commit()
    cur.close()
    del cur
    conn.close()

    print(f"\rSimulating {sim_name} is now completed!")


def panel_g_expert_times(save: bool = True, verbose: bool = False) -> None:
    simulation_types = [ids.SIM_CONVEX_OFC, ids.SIM_OFC_XAP_SWITCH]
    n_sims = len(simulation_types)
    db_paths = [get_path(simulation_type=s) for s in simulation_types]
    cols = [f"{keys.PERFORMANCE}_{t}" for t in range(constants.NR_TRIALS)]
    xp_t = {}
    for sim_t in simulation_types:
        xp_t[sim_t] = {}
        for ts in range(NR_CONTEXTS):
            xp_t[sim_t][ts] = np.zeros(constants.NR_SEEDS)

    helper.set_style()
    fig = plt.figure(figsize=(helper.PANEL_WIDTH, 1.35 * helper.PANEL_HEIGHT))
    ax = plt.gca()
    for s in range(n_sims):
        for t in range(NR_CONTEXTS):
            outcomes = np.array(sql.select(db_path=db_paths[s], table=get_table(TABLES[0], t), get_cols=cols))
            for seed in range(constants.NR_SEEDS):
                perf = ofcsst.utils.process.get_performance(outcomes[seed, :])
                xp_t[simulation_types[s]][t][seed] = ofcsst.utils.process.get_expert_t(perf)

    # Plot boxes of trials when expertise was reached
    colors = [COLOR_SINGLE, helper.COLOR_CONTEXT]
    bp_props = {'vert': False,
                'widths': 0.8,
                'boxprops': dict(linewidth=helper.LINE_WIDTH, color='black'),
                'capprops': dict(linewidth=helper.LINE_WIDTH),
                'whiskerprops': dict(linewidth=helper.LINE_WIDTH),
                'flierprops': dict(marker='o', markersize=helper.MARKER_SIZE, linewidth=helper.LINE_WIDTH),
                'medianprops': dict(color='k', solid_capstyle='butt')}
    expert_times: list = [None] * n_sims
    legend_handles = [None for _ in range(n_sims)]
    for s in range(n_sims):
        expert_times[s] = [xp_t[simulation_types[s]][c] for c in range(NR_CONTEXTS)]
        bp_props['flierprops']['markerfacecolor'] = colors[s]
        bp = ax.boxplot(expert_times[s], positions=[c * 3 + s for c in range(NR_CONTEXTS)],
                        patch_artist=True, **bp_props)
        legend_handles[s] = bp["boxes"][0]
        for patch in bp['boxes']:
            patch.set_facecolor(colors[s])
        for median in bp['medians']:
            median.set(linewidth=helper.LINE_WIDTH)
            median.set_color('black')

    xlim = [-1, 450]
    xmax = 450
    for c in range(NR_CONTEXTS - 1):
        x = 2 + 3 * c
        ax.plot([-1, xmax], [x, x], **helper.STYLE_RULE_SWITCH)

    x = 435
    n_test = NR_CONTEXTS
    if verbose:
        print('\nTesting difference between expert times for single-context and two contexts:')
    for c in range(NR_CONTEXTS):
        pv = stats.ttest_ind(expert_times[0][c], expert_times[1][c])[1] * n_test
        if verbose:
            print(f'Reversal {c} had p-value {pv:.1e}')
        ax.plot([x, x], [c * 3 - 0.25, c * 3 + 1.25], **helper.STYLE_SIGNIFICANT)
        if pv < constants.SIGNIFICANCE_THRESHOLD:
            ax.text(x + xlim[1] * 0.0025, c * 3 + 0.5, s=helper.get_significance(pv), rotation=-90, **helper.FONT_SIGNIFICANT)
        else:
            ax.text(x + xlim[1] * 0.015, c * 3 + 0.5, rotation=-90, va='center', **helper.FONT_NON_SIGNIFICANT)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines.bottom.set_bounds((-1, xmax))
    ax.set_yticks([0.5 + c * 3 for c in range(NR_CONTEXTS)], list(range(NR_CONTEXTS)))
    ax.set_xticks([100, 200, 300, 400])
    plt.ylim([NR_CONTEXTS * 3 - 1, -1])
    plt.xlim(xlim)
    plt.ylabel("Nth reversal")
    plt.xlabel("Trials to expertise")
    ax.legend(legend_handles, ['Single context', 'Two contexts'], loc='lower center', bbox_to_anchor=(0.5, 0.95),
              framealpha=1, ncol=2, frameon=False)

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    fig.subplots_adjust(top=0.9)
    helper.set_panel_label(label="g", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="g_expert_times")


def run():
    print('Simulating for Figure 4g')
    run_multi_reversal(simulation_type=ids.SIM_OFC_XAP_SWITCH)
    run_multi_reversal(simulation_type=ids.SIM_CONVEX_OFC)


def plot(save: bool = True):
    panel_g_expert_times(save=save)
