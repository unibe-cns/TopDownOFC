import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ofcsst import simulation
from ofcsst.utils import ids, keys, paths, constants, sql
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import get_data

NAME = "fig_s5_fake_ofc"
TASK_ID = ids.BINARY_2VN
PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs5"
SAVE_PATH = paths.SUPPLEMENTARY_SIMULATION_DIR / f"{NAME}.db"
NO_OFC_SIM = ids.SIM_CONVEX_GAIN
REV_RESET_SIM = ids.SIM_CONVEX_GAIN_REV_RESET
FAKE_OFC_SIM = ids.SIM_CONVEX_FAKE_OFC
SIM_NAMES = {NO_OFC_SIM: 'Default', REV_RESET_SIM: 'Reset values', FAKE_OFC_SIM: 'Fake SST'}
TRIAL_NRS = (constants.NR_TRIALS, constants.NR_TRIALS)


def simulate_condition(sim_type: ids.SimulationType):

    # Initialize database variables
    if sim_type == NO_OFC_SIM:
        seeds = list(range(3 * constants.NR_SEEDS, 4 * constants.NR_SEEDS))
    elif sim_type == REV_RESET_SIM:
        seeds = list(range(4 * constants.NR_SEEDS, 5 * constants.NR_SEEDS))
    elif sim_type == FAKE_OFC_SIM:
        seeds = list(range(5 * constants.NR_SEEDS, 6 * constants.NR_SEEDS))
    else:
        raise NotImplementedError(sim_type)
    # Initializations to save final simulation outcomes
    save_keys = [str(keys.SEED)] + [f"{keys.PERFORMANCE}_{t}" for t in range(sum(TRIAL_NRS))]
    conn = sql.connect(db_path=SAVE_PATH)
    sql.drop_table(conn=conn, table_name=sim_type, verbose=False)
    sql.make_table(conn=conn, table_name=sim_type, col_names=save_keys, verbose=False)
    conn.close()
    insert_cmd = sql.get_insert_cmd(table=sim_type, col_keys=save_keys)

    scan_db_path = simulation.scan.get_db_path(scan_type=ids.ST_FINAL, simulation_type=NO_OFC_SIM, task_id=TASK_ID,
                                               non_stationary=False)

    select_cols = simulation.databases.get_unique_cols(simulation_type=sim_type, table=simulation.databases.SCAN_TABLE)
    params = sql.get_max(db_path=scan_db_path, table=simulation.databases.BEST_PARAM_TABLE, group_cols=[],
                         select_cols=select_cols, max_col=keys.PERFORMANCE, maxmin=True)[0]

    # Run final simulations and save performance for each task
    outcomes = np.zeros((constants.NR_SEEDS, sum(TRIAL_NRS)))
    for s in range(constants.NR_SEEDS):
        print(f"\rSimulating seed {s}/{constants.NR_SEEDS}...", end="")
        outcomes[s, :] = simulation.simulate.simulate_params(task_id=TASK_ID, simulation_type=sim_type,
                                                             seed=seeds[s], params=list(params), nr_contexts=2,
                                                             n_trials=TRIAL_NRS)
    print(f"\rSimulation of {sim_type} completed.")

    conn = sqlite3.connect(SAVE_PATH)
    cur = conn.cursor()
    for s in range(constants.NR_SEEDS):
        values = tuple([seeds[s]] + outcomes[s, :].tolist())
        cur.execute(insert_cmd, values)
    conn.commit()
    cur.close()
    del cur
    conn.close()


def plot(save: bool = True) -> None:

    # Basic initializations
    simulation_types = [NO_OFC_SIM, REV_RESET_SIM, FAKE_OFC_SIM]
    colors = [helper.COLOR_GAIN, helper.COLOR_FAKE_OFC, (89. / 255., 41. / 255., 105. / 255.)]
    n_sims = len(simulation_types)

    # Get data
    t1_trials, t2_trials, t1_xp_t, t2_xp_t, t1_perf, t2_perf = get_data(db_path=SAVE_PATH,
                                                                        tables=simulation_types,
                                                                        trial_nrs=TRIAL_NRS)
    t1_perf_mean = np.zeros((n_sims, len(t1_trials)))
    t2_perf_mean = np.zeros((n_sims, len(t2_trials)))
    for i in range(n_sims):
        t1_perf_mean[i, :] = np.mean(t1_perf[i], axis=0)
        t2_perf_mean[i, :] = np.mean(t2_perf[i], axis=0)

    # Init plot
    helper.set_style()
    height_scale = 1.
    miny0, max_y0 = 0, max(np.max(t1_perf), np.max(t2_perf))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(2 * helper.PANEL_WIDTH, height_scale * helper.PANEL_HEIGHT),
                            height_ratios=[4 * height_scale, 1])
    rect = patches.Rectangle(xy=(0, miny0), width=100, height=max_y0-miny0, linewidth=0, facecolor=colors[2], alpha=0.3)
    axs[0, 1].add_patch(rect)

    # Plot all performance traces
    for seed in range(constants.NR_SEEDS):
        for i in range(n_sims):
            axs[0, 0].plot(t1_trials, t1_perf[i][seed, :], color=colors[i], alpha=0.05)
            axs[0, 1].plot(t2_trials, t2_perf[i][seed, :], color=colors[i], alpha=0.05)

    # Plot mean performance traces
    for i in range(n_sims):
        axs[0, 0].plot(t1_trials, t1_perf_mean[i, :], color=colors[i], label=SIM_NAMES[simulation_types[i]],
                       linewidth=1.5 * helper.LINE_WIDTH)
        axs[0, 1].plot(t2_trials, t2_perf_mean[i, :], color=colors[i], label=SIM_NAMES[simulation_types[i]],
                       linewidth=1.5 * helper.LINE_WIDTH)

    # Plot some cosmetics
    rect_style = dict(height=1., linewidth=0, alpha=0.4, clip_on=False)
    for i in range(n_sims):
        rect = patches.Rectangle(xy=(t1_trials[0], i + 0.5), width=t1_trials[-1], facecolor=colors[i], **rect_style)
        axs[1, 0].add_patch(rect)
        rect = patches.Rectangle(xy=(t2_trials[0], i + 0.5), width=t2_trials[-1], facecolor=colors[i], **rect_style)
        axs[1, 1].add_patch(rect)
    axs[0, 0].text(constants.NR_TRIALS - 12, 25, rotation=90, **helper.FONT_RULE_SWITCH)
    axs[1, 1].plot([t2_trials[0], t2_trials[0]], [-3.875 * height_scale * n_sims, n_sims + 0.5], clip_on=False,
                   zorder=10, **helper.STYLE_RULE_SWITCH)
    rect = patches.Rectangle(xy=(0, 0), width=0, height=0, facecolor=colors[2], alpha=0.3, label='Fake SST active')
    axs[0, 0].add_patch(rect)
    for i in range(2):
        axs[0, i].plot([0, TRIAL_NRS[i] - 1], [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
                       label="Expert threshold", **helper.STYLE_EXPERT_PERF)

    # Plot box plots of time to reach expertise
    box_props = dict(linewidth=helper.LINE_WIDTH, color='black')
    whisker_props = dict(linewidth=helper.LINE_WIDTH)
    flier_props = dict(marker='o', markersize=helper.MARKER_SIZE, linewidth=helper.LINE_WIDTH)
    for i in range(len(simulation_types)):
        axs[1, 0].boxplot(t1_xp_t[~np.isnan(t1_xp_t[:, i]), i], positions=[i+1], vert=False, notch=True, widths=0.5,
                          boxprops=box_props, whiskerprops=whisker_props, flierprops=flier_props)
        if simulation_types[i] == NO_OFC_SIM:
            xp_ts = t2_xp_t[~np.isnan(t2_xp_t[:, i]), i]
            ys = np.full(shape=(len(xp_ts),), fill_value=i + 1)
            axs[1, 1].scatter(xp_ts, ys, s=1.3 * helper.MARKER_SIZE, edgecolor='k', marker='o', facecolors='none')
        else:
            axs[1, 1].boxplot(t2_xp_t[~np.isnan(t2_xp_t[:, i]), i], positions=[i+1], vert=False, notch=True, widths=0.5,
                              boxprops=box_props, whiskerprops=whisker_props, flierprops=flier_props)

    # Legend and labels
    axs[0, 0].set_ylabel("Performance")
    fig.supxlabel("Trials (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.55, y=0.03)
    axs[0, 0].legend(bbox_to_anchor=(0.4, 0.68), frameon=False)

    # Set axes
    for a in range(2):
        axs[a, 0].set_xlim([t1_trials[0], t1_trials[-1]])
        axs[a, 1].set_xlim([t2_trials[0], t2_trials[-1]])
        axs[0, a].set_ylim([miny0, max_y0])
        axs[1, a].set_ylim([0.5, n_sims + 0.5])
        axs[1, a].invert_yaxis()
        axs[0, a].set_xticks([])
        axs[1, a].set_xticks([0, 100, 200, 300, 400, 500], ['', '100', '', '300', '', '500'])
        axs[1, a].set_yticks([])
    axs[0, 0].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[0, 1].spines[['bottom', 'left', 'right', 'top']].set_visible(False)
    axs[1, 0].spines[['right', 'top']].set_visible(False)
    axs[1, 1].spines[['left', 'right', 'top']].set_visible(False)
    axs[0, 0].spines['left'].set_bounds(miny0, 100)
    axs[0, 0].set_yticks([0, 100])
    axs[0, 1].set_yticks([])

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0, hspace=0)
    fig.subplots_adjust(bottom=0.2)
    helper.set_panel_label(label="b", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="b_perf_traces", plot_dpi=500, png=True)


def run():
    print('Running simulations for Figure S5')
    simulate_condition(sim_type=NO_OFC_SIM)
    simulate_condition(sim_type=REV_RESET_SIM)
    simulate_condition(sim_type=FAKE_OFC_SIM)
