import numpy as np
import matplotlib.pyplot as plt
import ofcsst.utils.process
from ofcsst.figures.helper import COLOR_CONTEXT
from ofcsst.simulation import databases, simulate
from ofcsst.simulation.scan import get_db_path, find_best_params
from ofcsst.utils import constants, ids, keys, paths, sql
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import get_data

TASK_ID = ids.BINARY_2VN
NAME = "fig_s4_reversal_no_distractor"
PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs4"
SAVE_PATH = paths.SUPPLEMENTARY_SIMULATION_DIR / f"{NAME}.db"
TRIAL_NRS = (constants.NR_TRIALS, 2 * constants.NR_TRIALS)
COR_PG = (195. / 255., 70. / 255., 108. / 255.)


def final_simulation(sim_type: ids.SimulationType, non_stationary_scan: bool) -> None:

    # Initializations to save final simulation outcomes
    seeds = list((range(constants.NR_SEEDS, 2 * constants.NR_SEEDS)))
    save_keys = [str(keys.SEED)] + [f"{keys.PERFORMANCE}_{t}" for t in range(sum(TRIAL_NRS))]
    conn = sql.connect(db_path=SAVE_PATH)
    table_name = f'{sim_type}_{non_stationary_scan}'
    sql.drop_table(conn=conn, table_name=table_name, verbose=False)
    sql.make_table(conn=conn, table_name=table_name, col_names=save_keys, verbose=False)
    conn.close()
    insert_cmd = sql.get_insert_cmd(table=table_name, col_keys=save_keys)

    scan_db_path = get_db_path(scan_type=ids.ST_NO_DISTRACTOR, simulation_type=sim_type, task_id=TASK_ID,
                               non_stationary=non_stationary_scan)
    select_cols = databases.get_unique_cols(simulation_type=sim_type, table=databases.SCAN_TABLE)
    params = sql.get_max(db_path=scan_db_path, table=databases.BEST_PARAM_TABLE, group_cols=[],
                         select_cols=select_cols, max_col=keys.PERFORMANCE, maxmin=True)[0]

    # Run final simulations and save performance for each task
    outcomes = np.zeros((constants.NR_SEEDS, sum(TRIAL_NRS)))
    for s in range(constants.NR_SEEDS):
        print(f'\rSimulating seed {s}/{constants.NR_SEEDS}...', end='')
        outcomes[s, :] = simulate.simulate_params(task_id=TASK_ID, simulation_type=sim_type, seed=seeds[s],
                                                  params=list(params), nr_contexts=2, n_trials=TRIAL_NRS)
    print(f'\rSimulating {sim_type} (nonstationary scan = {non_stationary_scan}) is finished.')
    conn = sql.connect(SAVE_PATH)
    cur = conn.cursor()
    for s in range(constants.NR_SEEDS):
        values = tuple([seeds[s]] + outcomes[s, :].tolist())
        cur.execute(insert_cmd, values)
    conn.commit()
    cur.close()
    del cur
    conn.close()


def scan():
    find_best_params(simulation_type=ids.SIM_CONVEX_NO_GAIN, task_id=ids.BINARY_2VN, switch_task=False,
                     scan_type=ids.ST_NO_DISTRACTOR)
    find_best_params(simulation_type=ids.SIM_CONVEX_NO_GAIN, task_id=ids.BINARY_2VN, switch_task=True,
                     scan_type=ids.ST_NO_DISTRACTOR)
    find_best_params(simulation_type=ids.SIM_PG_NO_GAIN, task_id=ids.BINARY_2VN, switch_task=False,
                     scan_type=ids.ST_NO_DISTRACTOR)
    find_best_params(simulation_type=ids.SIM_PG_NO_GAIN, task_id=ids.BINARY_2VN, switch_task=True,
                     scan_type=ids.ST_NO_DISTRACTOR)


def run():
    print('Running simulations for Figure S4')
    for non_stationary_scan in [False, True]:
        for sim_type in [ids.SIM_PG_NO_GAIN, ids.SIM_CONVEX_NO_GAIN]:
            final_simulation(sim_type=sim_type, non_stationary_scan=non_stationary_scan)


def panel_b_pyramid(save: bool = True) -> None:

    # Basic initializations
    simulation_types = [ids.SIM_PG_NO_GAIN, ids.SIM_CONVEX_NO_GAIN]
    colors = [COR_PG, (0.5, 0.5, 0.5)]

    # Database and columns containing data
    get_cols = [f"{keys.PERFORMANCE}_{t}" for t in range(sum(TRIAL_NRS))]

    # Plot expert time pyramid
    helper.set_style()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT),
                            height_ratios=TRIAL_NRS)
    t2_xp_t = [None, None]
    x_max = 105
    bin_width = 25
    for t in [0, 1]:
        n_trials = TRIAL_NRS[t]
        n_bins = n_trials // bin_width
        for i in range(len(simulation_types)):
            table = f'{simulation_types[i]}_{True}'
            if t:
                outcomes = np.array(
                    sql.select(db_path=SAVE_PATH, table=table, get_cols=get_cols))[:, TRIAL_NRS[0]:]
            else:
                outcomes = np.array(
                    sql.select(db_path=SAVE_PATH, table=table, get_cols=get_cols))[:, :TRIAL_NRS[0]]
            nr_seeds = outcomes.shape[0]
            t2_xp_t[i] = np.zeros((nr_seeds,))
            for seed in range(nr_seeds):
                performances = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, :])
                t2_xp_t[i][seed] = ofcsst.utils.process.get_expert_t(performances=performances)

            # Extract cumulative distribution
            histo = np.histogram(t2_xp_t[i], bins=n_bins, range=(0, n_trials))
            xs = histo[1][1:] - 0.5 * TRIAL_NRS[t] / n_bins
            cumulative_dist = 100*np.cumsum(histo[0]) / nr_seeds

            # Plot pyramid
            axs[t, i].barh(xs, cumulative_dist, color=colors[i], height=bin_width, linewidth=0.5, edgecolor="k")

            # Set axes
            axs[t, i].set_xlim([0, x_max])
            axs[t, i].set_ylim([0, TRIAL_NRS[t]])
            axs[t, i].spines[['right', 'top']].set_visible(False)
            axs[t, i].invert_yaxis()

        # More axes cosmetics
        axs[t, 0].invert_xaxis()
        axs[t, 1].set_yticks([])
        axs[0, t].set_xticks([])

    # Rule switch cosmetics
    axs[1, 1].plot([-x_max, x_max], [-90, -90], clip_on=False, **helper.STYLE_RULE_SWITCH)
    axs[1, 0].text(50, 0, clip_on=False, **helper.FONT_RULE_SWITCH)

    # More axes cosmetics
    axs[0, 0].set_yticks([1, 300, 600])
    axs[1, 0].set_yticks([1] + [300 * n for n in range(1, 5)])
    fig.supylabel("Trials (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.04, y=0.55)
    axs[1, 0].set_xlabel('secret label', alpha=0)
    fig.align_labels()

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0., hspace=0.2)
    fig.subplots_adjust(left=0.2)
    helper.set_panel_label(label="b", fig=fig)

    # Last labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Fraction experts [%]")
    axs[0, 0].text(50, -50, "PolGrad", horizontalalignment='center')
    axs[0, 1].text(50, -50, "Convex", horizontalalignment='center')

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="b_expert_time_pyramid")


def panel_c_2dxt(save: bool = True) -> None:

    # Basic initializations
    simulation_types = [ids.SIM_PG_NO_GAIN, ids.SIM_CONVEX_NO_GAIN]
    labels = ["PolGrad", "Convex"]
    tables = []
    colors = [COR_PG, (0.5, 0.5, 0.5)]
    colors_m = [COR_PG, 'k']
    for simulation_type in simulation_types:
        tables += [f'{simulation_type}_{True}']
    n_tables = len(tables)

    # Get data
    _, _, t1_xp_t, t2_xp_t, _, _ = get_data(db_path=SAVE_PATH, tables=tables, trial_nrs=TRIAL_NRS)
    helper.set_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))
    for t in range(n_tables):
        ax.scatter(t1_xp_t[:, t], t2_xp_t[:, t], marker='.', s=3, alpha=0.5, color=colors[t])
    for t in range(n_tables):
        x_mean, x_std = np.nanmean(t1_xp_t[:, t]), np.nanstd(t1_xp_t[:, t])
        y_mean, y_std = np.nanmean(t2_xp_t[:, t]), np.nanstd(t2_xp_t[:, t])
        ax.errorbar(x=x_mean, y=y_mean, xerr=x_std, yerr=y_std, color=colors_m[t], capsize=1, label=labels[t])
    ax.set_xlabel("Trials (learning)")
    ax.set_ylabel("Trials (reversal)")
    ax.set_xlim([0, 350])
    ax.set_ylim([0, 1100])
    ax.legend(frameon=False, loc='lower right')

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0)
    helper.set_panel_label(label="c", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="c_time_to_expert", plot_dpi=500)


def plot(save: bool = True):
    panel_b_pyramid(save=save)
    panel_c_2dxt(save=save)
