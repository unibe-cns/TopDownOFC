import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
import ofcsst.utils.process
from ofcsst.utils import constants, ids, keys, paths, sql
from ofcsst import simulation
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import OFC_SCAN, OFC_SIM, TASK_ID

NAME = 'fig_5_ofc_vip_ko'
PLOT_DIR = paths.MAIN_FIG_DIR / "fig5"
TRIAL_NRS = (1600, 1600)
TABLES = ['Forward', 'Reversed']


def get_path(sim_type: ids.SimulationType):
    return paths.RESULT_DIR / paths.SIMULATION_SUBDIR / f"{NAME}_{sim_type}.db"


def final_simulation(sim_type: ids.SimulationType, seeds: [int]) -> None:

    # Initializations to save final simulation outcomes
    save_keys = [[str(keys.SEED)] + [f"{keys.PERFORMANCE}_{t}" for t in range(TRIAL_NRS[0])],
                 [str(keys.SEED)] + [f"{keys.PERFORMANCE}_{t}" for t in range(TRIAL_NRS[1])]]
    save_db_path = get_path(sim_type=sim_type)
    conn = sql.connect(db_path=save_db_path)
    for t in range(len(TABLES)):
        sql.drop_table(conn=conn, table_name=TABLES[t], verbose=False)
        sql.make_table(conn=conn, table_name=TABLES[t], col_names=save_keys[t], verbose=False)
    conn.close()
    insert_cmds = [sql.get_insert_cmd(table=TABLES[t], col_keys=save_keys[t]) for t in range(len(TABLES))]

    scan_db_path = simulation.scan.get_db_path(scan_type=OFC_SCAN, simulation_type=OFC_SIM,
                                               task_id=TASK_ID, non_stationary=True)

    select_cols = simulation.databases.get_unique_cols(simulation_type=sim_type, table=simulation.databases.SCAN_TABLE)
    params = sql.get_max(db_path=scan_db_path, table=simulation.databases.BEST_PARAM_TABLE, group_cols=[],
                         select_cols=select_cols, max_col=keys.PERFORMANCE, maxmin=True)[0]

    # Run final simulations and save performance for each task
    outcomes = np.zeros((constants.NR_SEEDS, sum(TRIAL_NRS)))
    for s in range(constants.NR_SEEDS):
        print(f'\rFigure 5: Simulating seed {s}/{constants.NR_SEEDS}...', end='')
        outcomes[s, :] = simulation.simulate.simulate_params(task_id=TASK_ID, simulation_type=sim_type, seed=seeds[s],
                                                             params=list(params), nr_contexts=2, n_trials=TRIAL_NRS)
    print(f'\rFigure 5: Simulating {sim_type} finished successfully!')

    conn = sql.connect(save_db_path)
    cur = conn.cursor()
    trials = [np.array(list(range(TRIAL_NRS[0]))), np.array(list(range(TRIAL_NRS[0], sum(TRIAL_NRS))))]
    for s in range(constants.NR_SEEDS):
        for i in range(2):
            cur.execute(insert_cmds[i], tuple([seeds[s]] + outcomes[s, trials[i]].tolist()))
    conn.commit()
    cur.close()
    del cur
    conn.close()


def run():
    for sim_type in [OFC_SIM, ids.SIM_CONVEX_OFC_VIP_KO]:
        if sim_type == OFC_SIM:
            seeds = list(range(2 * constants.NR_SEEDS, 3 * constants.NR_SEEDS))

        elif sim_type == ids.SIM_CONVEX_OFC_VIP_KO:
            seeds = list(range(constants.NR_SEEDS, 2 * constants.NR_SEEDS))

        else:
            raise NotImplementedError(sim_type)

        final_simulation(sim_type=sim_type, seeds=seeds)


def plot(save: bool = True):

    print('Simulating for Figure 5a')

    # Init plot
    color = [helper.COLOR_DEFAULT, helper.COLOR_VIP]
    idxs = [0, 1]
    helper.set_style()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(1.3 * helper.PANEL_WIDTH, 1.2 * helper.PANEL_HEIGHT),
                            height_ratios=(9, 7))

    # Plot experimental data
    x_max = 17
    ylim = [-2, 6.]
    expert_thresh = 1.5
    xm = [15.9, 17]
    for i in idxs:
        axs[0, i].plot([1, xm[i]], [expert_thresh, expert_thresh], **helper.STYLE_EXPERT_PERF)

    # Cosmetics
    axs[0, 0].text(16.5, 3.5, rotation=90, **helper.FONT_RULE_SWITCH)
    axs[0, 1].plot([1, 1], ylim, clip_on=False, **helper.STYLE_RULE_SWITCH)
    for i in idxs:
        axs[0, i].set_xlim([1, x_max])
        axs[0, i].set_xticks([5, 10, 15])
        axs[0, i].set_ylim(ylim)
    axs[0, 0].set_yticks([-2, 0, 2, 4, 6])
    ne_x, ne_y = 2,  ylim[0] + 1.22 * (ylim[1] - ylim[0])
    axs[0, 0].text(1 + ne_x, ne_y, 'LN', ha='center', color=helper.COLOR_LN, weight='bold')
    axs[0, 0].text(x_max - ne_x, ne_y, 'LE', ha='center', color=helper.COLOR_LE, weight='bold')
    axs[0, 1].text(1 + ne_x, ne_y, 'RN', ha='center', color=helper.COLOR_RN, weight='bold')
    axs[0, 1].text(x_max - ne_x, ne_y, 'RE', ha='center', color=helper.COLOR_RE, weight='bold')

    # Get simulated data
    idxs = [0, 1]
    sims = [OFC_SIM, ids.SIM_CONVEX_OFC_VIP_KO]
    t1_trials = list(range(TRIAL_NRS[0]))
    t2_trials = list(range(TRIAL_NRS[1]))
    t1_perf = [np.zeros((constants.NR_SEEDS, TRIAL_NRS[0])) for _ in range(2)]
    t2_perf = [np.zeros((constants.NR_SEEDS, TRIAL_NRS[1])) for _ in range(2)]
    get_cols = [[f"{keys.PERFORMANCE}_{t}" for t in range(tn)] for tn in TRIAL_NRS]
    for i in idxs:

        # Forward task
        outcomes = np.array(sql.select(db_path=get_path(sim_type=sims[i]), table=TABLES[0], get_cols=get_cols[0]))
        for seed in range(constants.NR_SEEDS):
            t1_perf[i][seed, :] = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, :])

        # Reversed task
        outcomes = np.array(sql.select(db_path=get_path(sim_type=sims[i]), table=TABLES[1], get_cols=get_cols[1]))
        for seed in range(constants.NR_SEEDS):
            t2_perf[i][seed, :] = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, :])

    # Compute mean traces
    t1_perf_mean = np.zeros((2, len(t1_trials)))
    t2_perf_mean = np.zeros((2, len(t2_trials)))
    for i in idxs:
        t1_perf_mean[i, :] = np.mean(t1_perf[i], axis=0)
        t2_perf_mean[i, :] = np.mean(t2_perf[i], axis=0)

    # Plot simulated data
    y_min = 25
    labels = ['Control', 'CL=0']
    axs[1, 1].plot([0, 0], [y_min, 100], clip_on=False, **helper.STYLE_RULE_SWITCH)
    for i in idxs:
        axs[1, i].plot([0, TRIAL_NRS[i]], [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
                       **helper.STYLE_EXPERT_PERF)
        for seed in range(constants.NR_SEEDS):
            axs[1, 0].plot(t1_trials, t1_perf[i][seed, :], color=color[i], alpha=0.05)
            axs[1, 1].plot(t2_trials, t2_perf[i][seed, :], color=color[i], alpha=0.05)

    for i in idxs:
        axs[1, 0].plot(t1_trials, t1_perf_mean[i, :], color=color[i], label=labels[i])
        axs[1, 1].plot(t2_trials, t2_perf_mean[i, :], color=color[i], label=labels[i])
    ylim = list(axs[1, 1].get_ylim())
    ylim[0] = y_min
    for i in idxs:
        # axs[1, i].set_xlabel(f"Trials ({phase[i]})", labelpad=label_pad)
        axs[1, i].set_xlim([0, TRIAL_NRS[1]])
        axs[1, i].set_xticks([500, 1000, 1500])
        axs[1, i].set_ylim(ylim)
    fig.text(x=0.53, y=0.47, s="Sessions (after start/reversal)", fontsize=helper.FONT_SIZE, ha='center')
    fig.supxlabel("Trials (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.53, y=0.03)

    # Plot cosmetics
    rect = patches.Rectangle(xy=(0., 0.), width=1, height=0.15, linewidth=0, clip_on=False, zorder=-1,
                             facecolor=helper.COLOR_CNO, transform=axs[0, 1].transAxes)
    axs[0, 1].add_patch(rect)
    axs[0, 1].text(0.5, 0.06, '+CNO', transform=axs[0, 1].transAxes, fontsize=6, ha='center', va='center')
    axs[1, 0].legend(loc='lower center', handlelength=0.7, frameon=False)
    for k in range(2):
        axs[k, 1].get_yaxis().set_ticks([])
        axs[k, 0].spines[['right', 'top']].set_visible(False)
        axs[k, 1].spines[['left', 'right', 'top']].set_visible(False)
        for i in idxs:
            axs[k, i].tick_params(pad=2)
    axs[1, 0].spines['left'].set_bounds(y_min, 100)
    axs[0, 0].set_ylabel("Discrim. d'", labelpad=-1, loc='center')
    axs[1, 0].set_ylabel("Performance", labelpad=-1, loc='bottom')
    axs[1, 0].set_yticks([50, 100])
    axs[1, 1].set_yticks([])
    fig.align_ylabels(axs[:, 0])

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.5, wspace=0)
    fig.subplots_adjust(left=0.13, bottom=0.15, top=0.92)
    helper.set_panel_label(label="a", fig=fig)

    # Add learning colorbar
    normalize = mcolors.Normalize(vmin=0, vmax=1)
    x, y = 3, 6.8
    n_trials = 18
    naive_expert_style = {'ha': 'center', 'color': 'white', 'va': 'center'}
    phases = ['learning', 'reversal']
    for i in idxs:
        cmap = helper.get_performance_cmap(phase=phases[i])
        scalar_map = ScalarMappable(norm=normalize, cmap=cmap)
        cb = plt.colorbar(scalar_map, ax=axs[0, i], orientation='horizontal', location='top', ticks=[], drawedges=False,
                          fraction=0.16, aspect=11, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.zorder = -1
        axs[0, i].text(x, y, 'Naïve', **naive_expert_style)
        axs[0, i].text(n_trials - x, y, 'Expert', **naive_expert_style)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="a_performance_vip_cl")
