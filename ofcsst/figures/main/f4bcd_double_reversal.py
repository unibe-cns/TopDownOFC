import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
from pathlib import Path
from typing import cast
from ofcsst.utils import ids, keys, paths, process, sql, constants
from ofcsst.simulation import simulate, scan, databases
from ofcsst.figures import helper, statistics

PLOT_DIR = paths.MAIN_FIG_DIR / "fig4"
NR_TRIALS = (constants.NR_TRIALS, constants.NR_TRIALS, constants.NR_TRIALS)
TABLES = [
    keys.PERFORMANCE,
    f"{keys.AGENT_GAIN}_T1",
    f"{keys.AGENT_GAIN}_T2",
    f"{keys.AGENT_POLICY_WEIGHT}_T1",
    f"{keys.AGENT_POLICY_WEIGHT}_T2",
]
NR_CONTEXTS = 3
SINGLE_C_SIM = ids.SIM_CONVEX_OFC
COLOR_SINGLE = helper.COLOR_DEFAULT
PHASES = ("Learning", "Reversal 1", "Reversal 2")
LEARNING_RULES: tuple[ids.LearningRule, ...] = (ids.LR_L, ids.LR_R, ids.LR_R2)
MOUSE_TYPE = ids.MT_WT


def get_path() -> Path:
    return paths.RESULT_DIR / paths.SIMULATION_SUBDIR / "fig_4_double_reversal.db"


def init_sim(simulation_type: ids.SimulationType):
    if simulation_type in [
        ids.SIM_CONVEX_OFC,
        ids.SIM_OFC_XAP_SWITCH,
    ]:
        scan_sim_type = ids.SIM_CONVEX_OFC
        scan_type = ids.ST_TD
    elif simulation_type in [ids.SIM_CONVEX_GAIN]:
        scan_sim_type = ids.SIM_CONVEX_GAIN
        scan_type = ids.ST_FINAL
    else:
        raise NotImplementedError

    if simulation_type == ids.SIM_CONVEX_OFC:
        sim_name = "single-context"
        seeds = list(range(2 * constants.NR_SEEDS, 3 * constants.NR_SEEDS))
    elif simulation_type in [ids.SIM_OFC_XAP_SWITCH]:
        sim_name = "ofc gated xap switch"
        seeds = list(range(3 * constants.NR_SEEDS, 4 * constants.NR_SEEDS))
    else:
        raise NotImplementedError

    # Get the best parameters for each type of simulation
    scan_db_path = scan.get_db_path(
        scan_type=scan_type,
        simulation_type=scan_sim_type,
        task_id=constants.TASK_ID,
        non_stationary=True,
    )
    group_cols = [keys.NR_DISTRACTOR]
    select_cols = databases.get_unique_cols(
        simulation_type=scan_sim_type, table=databases.SCAN_TABLE
    )
    params: list = list(
        sql.get_max(
            db_path=scan_db_path,
            table=databases.BEST_PARAM_TABLE,
            group_cols=group_cols,
            select_cols=select_cols,
            max_col=keys.PERFORMANCE,
            maxmin=True,
        )[0]
    )

    return scan_sim_type, scan_type, sim_name, seeds, params


def get_cols(n_trials: int, with_seed: bool = True) -> dict:
    if with_seed:
        return {
            TABLES[0]: [str(keys.SEED)]
            + [f"{keys.PERFORMANCE}_{t}" for t in range(n_trials)],
            TABLES[1]: [str(keys.SEED)]
            + [f"{keys.AGENT_GAIN}_{t}" for t in range(n_trials)],
            TABLES[2]: [str(keys.SEED)]
            + [f"{keys.AGENT_GAIN}_{t}" for t in range(n_trials)],
            TABLES[3]: [str(keys.SEED)]
            + [f"{keys.AGENT_POLICY_WEIGHT}_{t}" for t in range(n_trials)],
            TABLES[4]: [str(keys.SEED)]
            + [f"{keys.AGENT_POLICY_WEIGHT}_{t}" for t in range(n_trials)],
        }
    else:
        return {
            TABLES[0]: [f"{keys.PERFORMANCE}_{t}" for t in range(n_trials)],
            TABLES[1]: [f"{keys.AGENT_GAIN}_{t}" for t in range(n_trials)],
            TABLES[2]: [f"{keys.AGENT_GAIN}_{t}" for t in range(n_trials)],
            TABLES[3]: [f"{keys.AGENT_POLICY_WEIGHT}_{t}" for t in range(n_trials)],
            TABLES[4]: [f"{keys.AGENT_POLICY_WEIGHT}_{t}" for t in range(n_trials)],
        }


def simulate_multi_reversal_analysis(
    simulation_type: ids.SimulationType,
    nr_reversals: int = 2,
    save_db_path: Path | None = None,
):
    # Initializations
    nr_contexts = nr_reversals + 1
    scan_sim_type, scan_type, sim_name, seeds, params = init_sim(
        simulation_type=simulation_type
    )
    col_keys = get_cols(n_trials=constants.NR_TRIALS * nr_contexts, with_seed=True)
    if save_db_path is None:
        save_db_path = get_path()
    conn = sql.connect(db_path=save_db_path)
    cur = conn.cursor()
    insert_cmd = {}
    for table in TABLES:
        sql.drop_table(conn=conn, table_name=table, verbose=False)
        sql.make_table(
            conn=conn, table_name=table, col_names=col_keys[table], verbose=False
        )
        insert_cmd[table] = sql.get_insert_cmd(table=table, col_keys=col_keys[table])

    # Run simulations for each seed
    for s in range(constants.NR_SEEDS):
        print(
            f"\rSimulating {sim_name} {100 * s / constants.NR_SEEDS:0.1f}% completed",
            end="",
        )

        # Simulate seed
        rewards, logger = simulate.run_simulation(
            task_id=constants.TASK_ID,
            simulation_type=simulation_type,
            params=params,
            seed=seeds[s],
            number_contexts=nr_contexts,
            log_type=ids.LT_AGENT,
        )
        if logger is None:
            raise ValueError(
                f"Logger is None for seed {seeds[s]} in simulation {sim_name}"
            )
        elif len(logger) != 5:
            raise ValueError(
                f"Logger has incorrect length for seed {seeds[s]} in simulation {sim_name}"
            )
        else:
            (gain_t1, gain_t2, policy_weights_t1, policy_weights_t2, _) = logger

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
    conn.close()

    print(f"\rSimulating {sim_name} is now completed!")


def panel_b_perf_xp(save: bool = True) -> None:
    """
    Plot performance traces of mice in experiments
    """

    # Extract highest number of sessions per learning rule
    max_sessions = 0
    for r in range(3):
        perf_ds = process.get_mouse_performances(
            mouse_type=MOUSE_TYPE, learning_rule=LEARNING_RULES[r]
        )
        max_sessions = max(max_sessions, perf_ds.shape[1])

    # Init figure
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(helper.PANEL_WIDTH, 0.46 * helper.PANEL_HEIGHT),
        squeeze=False,
    )
    fig = cast(plt.Figure, fig)
    axs = axs[0, 0]
    axs.plot(
        [0, 3 * max_sessions],
        [constants.EXPERT_D, constants.EXPERT_D],
        **helper.STYLE_EXPERT_PERF,
    )

    # Plot performance traces of mice in experiments
    for r in range(3):
        xs = list(range(r * max_sessions + 1, (r + 1) * max_sessions + 1))
        perf_ds = process.get_mouse_performances(
            mouse_type=MOUSE_TYPE, learning_rule=LEARNING_RULES[r]
        )
        for m in range(perf_ds.shape[0]):
            d = perf_ds[m, :]
            d = d[d != np.nan]
            axs.plot(xs[: len(d)], d, "k", alpha=0.5)

    # Plot cosmetics
    axs.spines[["right", "top"]].set_visible(False)
    axs.spines["left"].set_bounds(-2.5, 2.5)
    axs.set_xlim((1, 3 * max_sessions))
    axs.set_yticks([-2.5, 0, 2.5])
    axs.set_xticks(
        [
            1,
            5,
            max_sessions + 1,
            max_sessions + 5,
            2 * max_sessions + 1,
            2 * max_sessions + 5,
        ]
    )
    axs.set_xticklabels(["1", "5", "1", "5", "1", "5"])
    axs.set_xlabel("Sessions (after start/reversal)", labelpad=2)
    axs.set_ylabel("d'", va="baseline")
    axs.yaxis.set_label_coords(-0.15, 0.37)
    axs.text(
        14,
        4.5,
        f"{PHASES[0]}            {PHASES[1]}         {PHASES[2]}",
        ha="center",
        fontsize=helper.FONT_SIZE,
    )
    for x in [max_sessions + 0.5, 2 * max_sessions + 0.5]:
        axs.plot(
            [x, x], [-2.5, 5.5], clip_on=False, zorder=2, **helper.STYLE_RULE_SWITCH
        )

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    fig.subplots_adjust(top=0.9, left=0.2, bottom=0.35)
    helper.set_panel_label(label="b", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="b_mice_perf")


def panel_c_traces(save: bool = True) -> None:
    # Initializations
    n_trials = constants.NR_TRIALS
    db_path = get_path()
    cols = get_cols(n_trials=sum(NR_TRIALS), with_seed=False)
    task_trials = list(range(3 * n_trials))
    context_trials = [
        list(range(i * n_trials, (i + 1) * n_trials)) for i in range(NR_CONTEXTS)
    ]
    temp_perfs = np.zeros((NR_CONTEXTS, constants.NR_SEEDS, n_trials))
    t_ranges = [
        np.array(list(range(r * n_trials, (r + 1) * n_trials)))
        for r in range(NR_CONTEXTS)
    ]
    rows = [None, 1, 1, 2, 2]
    color = "k"
    fill_color = [
        (247.0 / 255.0, 231.0 / 255.0, 153 / 255.0),
        (144.0 / 255.0, 182.0 / 255.0, 165.0 / 255.0),
    ]

    linestyle_original = "dashed"
    linestyle_reversed = "dotted"
    linestyles = [
        None,
        linestyle_original,
        linestyle_reversed,
        linestyle_original,
        linestyle_reversed,
    ]

    # Process performances
    task_switches = list(range(NR_CONTEXTS))
    outcomes = np.array(
        sql.select(db_path=db_path, table=TABLES[0], get_cols=cols[TABLES[0]])
    )
    for seed in range(constants.NR_SEEDS):
        for t in task_switches:
            temp_perfs[t, seed, :] = process.get_performance(
                outcomes=outcomes[seed, t_ranges[t]]
            )
    seed_performances = np.concatenate(
        [temp_perfs[t, :, :] for t in task_switches], axis=1
    )

    # Init plot
    n_rows = 3
    helper.set_style()
    with rc_context({"mathtext.fontset": "cm"}):
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=1,
            figsize=(helper.PANEL_WIDTH, 1.35 * helper.PANEL_HEIGHT),
            height_ratios=(0.7, 1, 1),
        )
    fig = cast(plt.Figure, fig)
    axs[0].plot(
        [0, 3 * n_trials],
        [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
        **helper.STYLE_EXPERT_PERF,
    )

    # Plot performance traces
    for c in range(NR_CONTEXTS):
        ts = context_trials[c]
        mv = np.mean(seed_performances[:, ts], axis=0)
        stderr = np.std(seed_performances[:, ts], axis=0)
        axs[0].fill_between(
            ts, mv - stderr, mv + stderr, linewidth=0, color=(0.7, 0.7, 0.7)
        )
        axs[0].plot(ts, mv, color=color)

    # Plot reversal line cosmetics
    ymin = np.min(temp_perfs)
    ymax = np.max(temp_perfs)
    y_lims = [[ymin, ymax], [-0.7, 12], [-0.1, 1.1]]
    axs[0].set_zorder(100)
    axs[2].set_zorder(100)
    for row in range(n_rows):
        axs[row].patch.set_alpha(0.0)
        axs[row].set_xlim([0, 3 * n_trials])
        axs[row].set_ylim(y_lims[row])
    axs[0].text(n_trials - 38, 30, rotation=90, **helper.FONT_RULE_SWITCH)
    axs[0].text(2 * n_trials - 38, 30, rotation=90, **helper.FONT_RULE_SWITCH)
    for x in [n_trials - 0.5, 2 * n_trials - 0.5]:
        axs[2].plot(
            [x, x],
            [y_lims[2][0], y_lims[2][1] * 3.3],
            clip_on=False,
            zorder=2,
            **helper.STYLE_RULE_SWITCH,
        )

    # Plot traces of SNR and pi weight
    labels: list = [None for _ in range(len(TABLES))]
    labels[-2] = "Original rule"
    labels[-1] = "Reversed rule"
    for table_id in range(1, len(TABLES)):
        values = np.array(
            sql.select(
                db_path=db_path, table=TABLES[table_id], get_cols=cols[TABLES[table_id]]
            )
        )
        mv = np.mean(values, axis=0)[task_trials]
        stderr = np.std(values, axis=0)[task_trials]
        axs[rows[table_id]].plot(
            task_trials,
            mv,
            linestyle=linestyles[table_id],
            color=color,
            label=labels[table_id],
        )
        axs[rows[table_id]].fill_between(
            task_trials,
            mv - stderr,
            mv + stderr,
            color=fill_color[rows[table_id] - 1],
            linewidth=0,
        )

    axs[0].spines[["bottom", "right", "top"]].set_visible(False)
    axs[1].spines[["bottom", "right", "top"]].set_visible(False)
    axs[2].spines[["right", "top"]].set_visible(False)
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([100 + 200 * i for i in range(9)], ["100", "300", "500"] * 3)
    axs[0].spines["left"].set_bounds(0, 100)
    axs[0].set_yticks([0, 100])
    axs[0].set_ylabel("Performance", va="baseline")
    axs[1].spines["left"].set_bounds(0, 10)
    axs[1].set_yticks([0, 10])
    axs[1].set_ylabel(r"Rel. gain $\Gamma_\mathrm{go}^\mathrm{ap}$", va="baseline")
    axs[2].spines["left"].set_bounds(y_lims[2][0], 1)
    axs[2].set_yticks([0, 1])
    axs[2].set_ylabel(r"$\pi$-weight", va="baseline")
    axs[2].set_xlabel("Trials (after start/reversal)")
    axs[0].set_title(
        "  Learning          Reversal 1        Reversal 2", fontsize=helper.FONT_SIZE
    )
    fig.align_ylabels(axs)
    for ax in axs:
        ax.yaxis.set_label_coords(-0.15, 0.5)

    # Plot cosmetics like legends and reversal lines
    leg = axs[2].legend(
        loc="lower left",
        framealpha=1,
        ncol=2,
        handlelength=2.4,
        bbox_to_anchor=(0.0, 0.82, 1.0, 0.5),
    )
    axs[2].add_artist(leg)
    lines = [
        plt.Line2D([], [], linestyle=ls, color="k")
        for ls in [linestyle_original, linestyle_reversed]
    ]
    with rc_context({"mathtext.fontset": "cm"}):
        labels = [
            r"$\Gamma_\mathrm{s_1}^\mathrm{ap}$",
            r"$\Gamma_\mathrm{s_2}^\mathrm{ap}$",
        ]
        legend_arg = {"loc": "lower right", "frameon": False, "handlelength": 1.7}
        axs[1].legend(
            handles=lines,
            labels=labels,
            bbox_to_anchor=(0.5, 0.1, 0.5, 0.5),
            **legend_arg,
        )
        labels = [r"$w_\mathrm{s_1}^\mathrm{PG}$", r"$w_\mathrm{s_2}^\mathrm{PG}$"]
        axs[2].legend(
            handles=lines,
            labels=labels,
            bbox_to_anchor=(0.5, -0.05, 0.5, 0.5),
            **legend_arg,
        )

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.1, wspace=0)
    fig.subplots_adjust(bottom=0.13, left=0.2)
    helper.set_panel_label(label="c", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="c_traces")


def panel_d_expert_times(save: bool = True, verbose: bool = False) -> None:
    # Init plot
    helper.set_style()
    fig_width, fig_height = 1.0 * helper.PANEL_WIDTH, 0.6 * helper.PANEL_HEIGHT
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height))
    fig = cast(plt.Figure, fig)
    xp_id, sim_id = 0, 1

    # Plot box-plots of sessions when expertise was reached by mice in experiments
    xp_t = {}
    perf_per_group = [np.empty((0,)) for _ in range(3)]
    for r in range(3):
        perf_ds = process.get_mouse_performances(
            mouse_type=MOUSE_TYPE, learning_rule=LEARNING_RULES[r]
        )
        perf_per_group[r] = perf_ds.copy()
        n_mice = perf_ds.shape[0]
        xp_t[r] = [0 for _ in range(n_mice)]
        for m in range(n_mice):
            d = perf_ds[m, :]
            d[d == np.nan] = 0
            xp_t[r][m] = 1 + int(np.argmax(d > constants.EXPERT_D))

    # Significance tests
    idxs = [[0, 1], [0, 2], [1, 2]]
    n_test = len(idxs)
    pvs = [
        statistics.logrank_cox_ph(
            perf_per_group[i[0]],
            perf_per_group[i[1]],
            n_tests=n_test,
            outcome="mouse learning time",
            label_a=PHASES[i[0]],
            label_b=PHASES[i[1]],
            verbose=verbose,
        )
        for i in idxs
    ]

    ref_x = 8.5
    xs = [ref_x, ref_x + 1, ref_x + 2]
    ys = [[1, 2], [1, 3], [2, 3]]
    for i in range(n_test):
        axs[xp_id].plot([xs[i], xs[i]], ys[i], **helper.STYLE_SIGNIFICANT)
        if pvs[i] < constants.SIGNIFICANCE_THRESHOLD:
            axs[xp_id].text(
                xs[i],
                sum(ys[i]) / 2.0,
                s=helper.get_significance(pvs[i]),
                rotation=270,
                **helper.FONT_SIGNIFICANT,
            )
        else:
            axs[xp_id].text(
                xs[i] + 0.35,
                sum(ys[i]) / 2.0,
                rotation=270,
                va="center",
                **helper.FONT_NON_SIGNIFICANT,
            )
    bp_props = {
        "vert": False,
        "widths": 0.5,
        "boxprops": dict(linewidth=helper.AXIS_WIDTH, color="black"),
        "capprops": dict(linewidth=helper.AXIS_WIDTH),
        "whiskerprops": dict(linewidth=helper.AXIS_WIDTH),
        "flierprops": dict(
            marker="o", markersize=helper.MARKER_SIZE, linewidth=helper.AXIS_WIDTH
        ),
        "medianprops": dict(
            color="k", solid_capstyle="butt", linewidth=helper.AXIS_WIDTH
        ),
        "showfliers": False,
    }

    for ts in range(len(xp_t)):
        axs[xp_id].boxplot(xp_t[ts], positions=[ts + 1], **bp_props)
        y = ts + np.linspace(0.8, 1.2, len(xp_t[ts]))
        axs[xp_id].scatter(xp_t[ts], y, **helper.BP_SCATTER)

    # Get trials when expertise was reached by model in simulations
    db_path = get_path()
    cols = [f"{keys.PERFORMANCE}_{t}" for t in range(sum(NR_TRIALS))]
    xp_t = {}
    for ts in range(NR_CONTEXTS + 1):
        xp_t[ts] = np.zeros(constants.NR_SEEDS)
    t_ranges = [
        np.array(list(range(r * constants.NR_TRIALS, (r + 1) * constants.NR_TRIALS)))
        for r in range(NR_CONTEXTS)
    ]
    outcomes = np.array(sql.select(db_path=db_path, table=TABLES[0], get_cols=cols))
    perfs = np.zeros((NR_CONTEXTS, constants.NR_SEEDS, constants.NR_TRIALS))
    for seed in range(constants.NR_SEEDS):
        for t in range(NR_CONTEXTS):
            perfs[t, seed, :] = process.get_performance(outcomes[seed, t_ranges[t]])
            xp_t[t][seed] = process.get_expert_t(perfs[t, seed, :])

    # Plot boxes of trials when expertise was reached by model in simulation
    bp_props["boxprops"]["facecolor"] = COLOR_SINGLE
    for ts in range(NR_CONTEXTS):
        xp_t[ts] = xp_t[ts][~np.isnan(xp_t[ts])]
        axs[sim_id].boxplot(
            np.transpose(xp_t[ts]), positions=[ts + 1], patch_artist=True, **bp_props
        )
        y = ts + np.linspace(0.8, 1.2, len(xp_t[ts]))
        axs[sim_id].scatter(xp_t[ts], y, **helper.BP_SCATTER)

    # Plot significance stars
    pvs = [
        statistics.logrank_cox_ph(
            perfs[i[0], :],
            perfs[i[1], :],
            n_tests=n_test,
            outcome="simulated learning time",
            label_a=PHASES[i[0]],
            label_b=PHASES[i[1]],
            verbose=verbose,
            simulation=True,
        )
        for i in idxs
    ]
    ref_x = constants.NR_TRIALS - 160
    xs = [ref_x + 10, ref_x + 120, ref_x + 65]
    ys = [[1, 2], [1, 3], [2, 3]]
    for i in range(n_test):
        axs[sim_id].plot([xs[i], xs[i]], ys[i], **helper.STYLE_SIGNIFICANT)
        if pvs[i] < constants.SIGNIFICANCE_THRESHOLD:
            axs[sim_id].text(
                xs[i],
                sum(ys[i]) / 2.0,
                s=helper.get_significance(pvs[i]),
                rotation=270,
                **helper.FONT_SIGNIFICANT,
            )
        else:
            axs[sim_id].text(
                xs[i], sum(ys[i]) / 2.0, rotation=270, **helper.FONT_NON_SIGNIFICANT
            )

    # Axis cosmetics
    xlim = [[], []]
    xlim[xp_id] = [1, 11]
    xlim[sim_id] = [0, constants.NR_TRIALS]
    for r in range(2):
        axs[r].invert_yaxis()
        axs[r].spines[["right", "top"]].set_visible(False)
        axs[r].tick_params(axis="both", which="major", pad=1)
        axs[r].set_xlim(xlim[r])
    axs[sim_id].set_xticks(
        [0, 100, 200, 300, 400, 500, 600], ["", "100", "", "300", "", "500", ""]
    )
    axs[xp_id].set_xticks(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ["1", "", "3", "", "5", "", "7", "", "9", "", "11"],
    )
    axs[sim_id].set_xlabel("Trials to expertise")
    axs[xp_id].set_xlabel("Sessions to expertise")
    axs[1].set_yticklabels([])
    axs[0].set_yticklabels(PHASES)

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0.1)
    fig.subplots_adjust(top=0.8)
    helper.set_panel_label(label="d", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="d_expertise")


def run():
    print("Simulating for Figure 4cd:")
    simulate_multi_reversal_analysis(simulation_type=SINGLE_C_SIM)


def plot(save: bool = True):
    panel_b_perf_xp(save=save)
    panel_c_traces(save=save)
    panel_d_expert_times(save=save)
