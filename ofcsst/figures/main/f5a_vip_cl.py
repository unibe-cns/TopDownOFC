import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from typing import cast
import ofcsst.utils.process
from ofcsst.utils import constants, ids, keys, paths, process, sql
from ofcsst import simulation
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import OFC_SIM, TASK_ID

NAME = "fig_5_ofc_vip_ko"
PLOT_DIR = paths.MAIN_FIG_DIR / "fig5"
TRIAL_NRS = (1600, 1600)
TABLES = ["Forward", "Reversed"]
MOUSE_TYPES = (ids.MT_VIP_WT, ids.MT_VIP_KO)
LEARNING_RULES = (ids.LR_L, ids.LR_R)


def get_path(sim_type: ids.SimulationType):
    return paths.RESULT_DIR / paths.SIMULATION_SUBDIR / f"{NAME}_{sim_type}.db"


def final_simulation(sim_type: ids.SimulationType, seeds: list[int]) -> None:
    # Initializations to save final simulation outcomes
    save_keys = [
        [str(keys.SEED)] + [f"{keys.PERFORMANCE}_{t}" for t in range(TRIAL_NRS[0])],
        [str(keys.SEED)] + [f"{keys.PERFORMANCE}_{t}" for t in range(TRIAL_NRS[1])],
    ]
    save_db_path = get_path(sim_type=sim_type)
    conn = sql.connect(db_path=save_db_path)
    for t, table in enumerate(TABLES):
        sql.drop_table(conn=conn, table_name=table, verbose=False)
        sql.make_table(
            conn=conn, table_name=table, col_names=save_keys[t], verbose=False
        )
    conn.close()
    insert_cmds = [
        sql.get_insert_cmd(table=table, col_keys=save_keys[t])
        for t, table in enumerate(TABLES)
    ]

    scan_db_path = simulation.scan.get_db_path(
        scan_type=ids.ST_TD,
        simulation_type=OFC_SIM,
        task_id=TASK_ID,
        non_stationary=True,
    )

    select_cols = simulation.databases.get_unique_cols(
        simulation_type=sim_type, table=simulation.databases.SCAN_TABLE
    )
    params = sql.get_max(
        db_path=scan_db_path,
        table=simulation.databases.BEST_PARAM_TABLE,
        group_cols=[],
        select_cols=select_cols,
        max_col=keys.PERFORMANCE,
        maxmin=True,
    )[0]

    # Run final simulations and save performance for each task
    outcomes = np.zeros((constants.NR_SEEDS, sum(TRIAL_NRS)))
    for s in range(constants.NR_SEEDS):
        print(f"\rFigure 5: Simulating seed {s}/{constants.NR_SEEDS}...", end="")
        outcomes[s, :] = simulation.simulate.simulate_params(
            task_id=TASK_ID,
            simulation_type=sim_type,
            seed=seeds[s],
            params=list(params),
            nr_contexts=2,
            n_trials=TRIAL_NRS,
        )
    print(f"\rFigure 5: Simulating {sim_type} finished successfully!")

    conn = sql.connect(save_db_path)
    cur = conn.cursor()
    trials = [
        np.array(list(range(TRIAL_NRS[0]))),
        np.array(list(range(TRIAL_NRS[0], sum(TRIAL_NRS)))),
    ]
    for s in range(constants.NR_SEEDS):
        for i in range(2):
            cur.execute(
                insert_cmds[i], tuple([seeds[s]] + outcomes[s, trials[i]].tolist())
            )
    conn.commit()
    cur.close()
    conn.close()


def run():
    print("Running simulations for Figure 5a")

    for sim_type in (OFC_SIM, ids.SIM_CONVEX_OFC_VIP_KO):
        if sim_type == OFC_SIM:
            seeds = list(range(2 * constants.NR_SEEDS, 3 * constants.NR_SEEDS))

        elif sim_type == ids.SIM_CONVEX_OFC_VIP_KO:
            seeds = list(range(constants.NR_SEEDS, 2 * constants.NR_SEEDS))

        else:
            raise NotImplementedError(sim_type)

        final_simulation(sim_type=sim_type, seeds=seeds)


def plot(save: bool = True):
    # Init plot
    color = [helper.COLOR_DEFAULT, helper.COLOR_VIP]
    phases = list(range(len(LEARNING_RULES)))
    conditions = list(range(len(color)))
    plot_rows = [0, 1]
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=len(plot_rows),
        ncols=len(phases),
        figsize=(1.3 * helper.PANEL_WIDTH, 1.2 * helper.PANEL_HEIGHT),
        height_ratios=(19, 16),
        squeeze=False,
    )
    fig = cast(plt.Figure, fig)
    sr, er = 0, 1
    y_lims = [[], []]
    x_maxs = [0, 0]

    # Get simulated data
    sims: list[ids.SimulationType] = [OFC_SIM, ids.SIM_CONVEX_OFC_VIP_KO]
    t1_trials = list(range(TRIAL_NRS[0]))
    t2_trials = list(range(TRIAL_NRS[1]))
    t1_perf = [np.zeros((constants.NR_SEEDS, TRIAL_NRS[0])) for _ in range(2)]
    t2_perf = [np.zeros((constants.NR_SEEDS, TRIAL_NRS[1])) for _ in range(2)]
    get_cols = [[f"{keys.PERFORMANCE}_{t}" for t in range(tn)] for tn in TRIAL_NRS]
    for i in conditions:
        # Forward task
        outcomes = np.array(
            sql.select(
                db_path=get_path(sim_type=sims[i]),
                table=TABLES[0],
                get_cols=get_cols[0],
            )
        )
        for seed in range(constants.NR_SEEDS):
            t1_perf[i][seed, :] = ofcsst.utils.process.get_performance(
                outcomes=outcomes[seed, :]
            )

        # Reversed task
        outcomes = np.array(
            sql.select(
                db_path=get_path(sim_type=sims[i]),
                table=TABLES[1],
                get_cols=get_cols[1],
            )
        )
        for seed in range(constants.NR_SEEDS):
            t2_perf[i][seed, :] = ofcsst.utils.process.get_performance(
                outcomes=outcomes[seed, :]
            )

    # Compute mean traces
    t1_perf_mean = np.zeros((2, len(t1_trials)))
    t2_perf_mean = np.zeros((2, len(t2_trials)))
    for i in conditions:
        t1_perf_mean[i, :] = np.mean(t1_perf[i], axis=0)
        t2_perf_mean[i, :] = np.mean(t2_perf[i], axis=0)

    # Plot simulated data
    y_min = 25
    x_maxs[sr] = max(TRIAL_NRS)
    labels = ["Control", "CL=0"]
    axs[sr, 1].plot([0, 0], [y_min, 100], clip_on=False, **helper.STYLE_RULE_SWITCH)
    for i in phases:
        axs[sr, i].plot(
            [0, TRIAL_NRS[i]],
            [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
            **helper.STYLE_EXPERT_PERF,
        )
        for seed in range(constants.NR_SEEDS):
            axs[sr, 0].plot(t1_trials, t1_perf[i][seed, :], color=color[i], alpha=0.05)
            axs[sr, 1].plot(t2_trials, t2_perf[i][seed, :], color=color[i], alpha=0.05)

    for i in conditions:
        axs[sr, 0].plot(t1_trials, t1_perf_mean[i, :], color=color[i], label=labels[i])
        axs[sr, 1].plot(t2_trials, t2_perf_mean[i, :], color=color[i], label=labels[i])
    y_lims[sr] = list(axs[sr, 1].get_ylim())
    y_lims[sr][0] = y_min
    for i in phases:
        axs[sr, i].set_xlim([0, x_maxs[sr]])
        axs[sr, i].set_xticks([500, 1000, 1500])
        axs[sr, i].set_ylim(y_lims[sr])

    # Plot experimental data
    y_lims[er] = [-2, 6.0]
    expert_thresh = 1.5
    label = ["WT C57 control", "VIP-Cre/FLEX\nhM4Di in lOFC"]
    xm = [15.9, 17]
    xp_t = {}
    data_lists = [[], []]
    ms = [[[], []], [[], []]]
    bs = [[[], []], [[], []]]
    for i in phases:
        axs[er, i].plot(
            [1, xm[i]], [expert_thresh, expert_thresh], **helper.STYLE_EXPERT_PERF
        )
        xp_t[i] = {}
        for j in conditions:
            data: np.ndarray = process.get_mouse_performances(
                mouse_type=MOUSE_TYPES[j], learning_rule=LEARNING_RULES[i]
            )
            x_maxs[er] = max(x_maxs[er], data.shape[1])
            mean_x = np.nanmean(data, axis=0)
            xs = np.linspace(1, len(mean_x), len(mean_x))
            axs[er, i].plot(xs, np.transpose(data), color=color[j], alpha=0.2)
            axs[er, i].plot(xs, mean_x, color=color[j], label=label[j])

            for k in range(data.shape[0]):
                performance = data[k, ~np.isnan(data[k, :])]
                m, b = np.polyfit(np.arange(len(performance)), performance, 1)
                ms[i][j].append(m)
                bs[i][j].append(b)
                for t, p in enumerate(performance):
                    data_lists[i].append(
                        {
                            "Subject": f"mouse_{k}",
                            "Group": f"condition_{j}",
                            "Time": t,
                            "Performance": p,
                        }
                    )
    for i in phases:
        axs[er, i].set_xlim([1, x_maxs[er]])
        axs[er, i].set_xticks([5, 10, 15])
        axs[er, i].set_ylim(y_lims[er])
    axs[er, 0].set_yticks([-2, 0, 2, 4, 6])
    axs[er, 0].text(
        0.973 * x_maxs[er],
        y_lims[er][0] + 0.68 * (y_lims[er][1] - y_lims[er][0]),
        rotation=90,
        **helper.FONT_RULE_SWITCH,
    )
    axs[er, 1].plot([1, 1], y_lims[er], clip_on=False, **helper.STYLE_RULE_SWITCH)

    # Learning stage cosmetics
    ne_x, ne_y = 0.117 * x_maxs[0], y_lims[0][0] + 1.19 * (y_lims[0][1] - y_lims[0][0])
    axs[0, 0].text(
        1 + ne_x, ne_y, "LN", ha="center", color=helper.COLOR_LN, weight="bold"
    )
    axs[0, 0].text(
        x_maxs[0] - ne_x, ne_y, "LE", ha="center", color=helper.COLOR_LE, weight="bold"
    )
    axs[0, 1].text(
        1 + ne_x, ne_y, "RN", ha="center", color=helper.COLOR_RN, weight="bold"
    )
    axs[0, 1].text(
        x_maxs[0] - ne_x, ne_y, "RE", ha="center", color=helper.COLOR_RE, weight="bold"
    )

    # X labels
    ss = ["Trials", "Sessions"]
    fig.text(
        x=0.53,
        y=0.478,
        s=f"{ss[sr]} (after start/reversal)",
        fontsize=helper.FONT_SIZE,
        ha="center",
    )
    fig.supxlabel(
        f"{ss[er]} (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.53, y=0.03
    )

    # CNO cosmetics
    rect = patches.Rectangle(
        xy=(0.0, 0.0),
        width=1,
        height=0.15,
        linewidth=0,
        clip_on=False,
        zorder=-1,
        facecolor=helper.COLOR_CNO,
        transform=axs[er, 1].transAxes,
    )
    axs[er, 1].add_patch(rect)
    axs[er, 1].text(
        0.5,
        0.06,
        "+CNO",
        transform=axs[er, 1].transAxes,
        fontsize=6,
        ha="center",
        va="center",
    )

    # Legends and axes
    axs[er, 1].legend(
        bbox_to_anchor=(0.36, 0.44), borderaxespad=0, handlelength=0.7, frameon=False
    )
    axs[sr, 0].legend(loc="lower center", handlelength=0.7, frameon=False)
    for k in plot_rows:
        axs[k, 1].get_yaxis().set_ticks([])
        axs[k, 0].spines[["right", "top"]].set_visible(False)
        axs[k, 1].spines[["left", "right", "top"]].set_visible(False)
        for i in phases:
            axs[k, i].tick_params(pad=2)
    axs[sr, 0].spines["left"].set_bounds(y_min, 100)
    axs[er, 0].set_ylabel("Discrim. d'", labelpad=-1)
    axs[sr, 0].set_ylabel("Performance  ", labelpad=-1)
    axs[sr, 0].set_yticks([50, 100])
    axs[sr, 1].set_yticks([])
    fig.align_ylabels(axs[:, 0])

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.5, wspace=0)
    fig.subplots_adjust(left=0.13, bottom=0.15, top=0.92)
    helper.set_panel_label(label="a", fig=fig)
    _, fig_height = fig.get_size_inches()
    fig.gca().text(
        helper.FIG_WPAD,
        0.95,
        "b",
        transform=fig.dpi_scale_trans,
        fontsize=8,
        fontweight="bold",
    )

    # Add learning stage colorbar
    normalize = mcolors.Normalize(vmin=0, vmax=1)
    naive_expert_style = {"ha": "center", "color": "white", "va": "center"}
    pads = [0.01, -0.02]
    for i in phases:
        cmap = helper.get_performance_cmap(phase=LEARNING_RULES[i])
        scalar_map = ScalarMappable(norm=normalize, cmap=cmap)
        cb = plt.colorbar(
            scalar_map,
            ax=axs[0, i],
            orientation="horizontal",
            location="top",
            ticks=[],
            drawedges=False,
            fraction=0.16,
            aspect=11,
            pad=pads[er],
        )
        cb.outline.set_visible(False)
        cb.ax.zorder = 0
        cb.ax.text(0.12, 0.5, "Naïve", **naive_expert_style)
        cb.ax.text(0.88, 0.5, "Expert", **naive_expert_style)

    # Save or display
    helper.save_or_show(
        save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="a_performance_vip_cl"
    )
