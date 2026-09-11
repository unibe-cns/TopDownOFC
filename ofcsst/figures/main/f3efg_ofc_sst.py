import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from pathlib import Path
from typing import Any, cast
from dataclasses import dataclass, field

from ofcsst import simulation
from ofcsst.utils import constants, ids, keys, paths, process, sql
from ofcsst.figures import helper, statistics

TASK_ID = ids.BINARY_2VN
NAME = "fig_3efg_ofc_sst"
PLOT_DIR = paths.MAIN_FIG_DIR / "fig3"
TRIAL_NRS = (constants.NR_TRIALS, 2 * constants.NR_TRIALS)
OFC_SIM = ids.SIM_CONVEX_OFC
NO_OFC_SIM = ids.SIM_CONVEX_GAIN
SIM_NAMES = {OFC_SIM: "Intact lOFC", NO_OFC_SIM: "Silenced lOFC"}
COLOR_WT = helper.COLOR_DEFAULT
COLOR_NO_OFC = helper.COLOR_SST
SAVE_PATH = paths.RESULT_DIR / paths.SIMULATION_SUBDIR / f"{NAME}.db"


def final_simulation(
    sim_type: ids.SimulationType,
    scan_path: Path,
    seeds: list[int],
    save_db_path: Path = SAVE_PATH,
) -> None:
    # Initializations to save final simulation outcomes
    n_seeds = len(seeds)
    save_keys = [str(keys.SEED)] + [
        f"{keys.PERFORMANCE}_{t}" for t in range(sum(TRIAL_NRS))
    ]
    conn = sql.connect(db_path=save_db_path)
    sql.drop_table(conn=conn, table_name=sim_type, verbose=False)
    sql.make_table(conn=conn, table_name=sim_type, col_names=save_keys, verbose=False)
    conn.close()
    insert_cmd = sql.get_insert_cmd(table=sim_type, col_keys=save_keys)

    select_cols = simulation.databases.get_unique_cols(
        simulation_type=sim_type, table=simulation.databases.SCAN_TABLE
    )
    params = sql.get_max(
        db_path=scan_path,
        table=simulation.databases.BEST_PARAM_TABLE,
        group_cols=[],
        select_cols=select_cols,
        max_col=keys.PERFORMANCE,
        maxmin=True,
    )[0]

    # Run final simulations and save performance for each task
    outcomes = np.zeros((n_seeds, sum(TRIAL_NRS)))
    for s in range(n_seeds):
        print(f"\rSimulating seed {s}/{n_seeds}...", end="")
        outcomes[s, :] = simulation.simulate.simulate_params(
            task_id=TASK_ID,
            simulation_type=sim_type,
            seed=seeds[s],
            params=list(params),
            nr_contexts=2,
            n_trials=TRIAL_NRS,
        )
    print(f"\rSimulating {sim_type} finished successfully!")

    conn = sqlite3.connect(save_db_path)
    cur = conn.cursor()
    for s in range(n_seeds):
        values = tuple([seeds[s]] + outcomes[s, :].tolist())
        cur.execute(insert_cmd, values)
    conn.commit()
    cur.close()
    conn.close()


def simulate_condition(simulation_type: ids.SimulationType):
    # Initialize database variables
    if simulation_type == NO_OFC_SIM:
        scan_type = ids.ST_FINAL
        seeds = list(range(constants.NR_SIM_MICE, 2 * constants.NR_SIM_MICE))

    elif simulation_type == OFC_SIM:
        scan_type = ids.ST_TD
        seeds = list(range(2 * constants.NR_SIM_MICE, 3 * constants.NR_SIM_MICE))

    else:
        raise NotImplementedError(simulation_type)
    scan_db_path = simulation.scan.get_db_path(
        simulation_type=simulation_type, scan_type=scan_type
    )

    final_simulation(sim_type=simulation_type, scan_path=scan_db_path, seeds=seeds)


def get_data(
    db_path: Path = SAVE_PATH,
    sim_types: tuple[ids.SimulationType, ...] | None = None,
    trial_nrs=TRIAL_NRS,
) -> tuple[
    list[int], list[int], np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]
]:
    # Database tables and columns to get
    sim_types = sim_types or (NO_OFC_SIM, OFC_SIM)
    get_cols = [f"{keys.PERFORMANCE}_{t}" for t in range(sum(trial_nrs))]

    # Object to store processed data
    t1_trials = list(range(trial_nrs[0]))
    t2_trials = list(range(trial_nrs[1]))
    t1_xp_t = np.zeros((constants.NR_SIM_MICE, len(sim_types)))
    t2_xp_t = np.zeros((constants.NR_SIM_MICE, len(sim_types)))
    t1_perf = [
        np.zeros((constants.NR_SIM_MICE, trial_nrs[0])) for _ in range(len(sim_types))
    ]
    t2_perf = [
        np.zeros((constants.NR_SIM_MICE, trial_nrs[1])) for _ in range(len(sim_types))
    ]

    # Get and process data for each condition
    for i in range(len(sim_types)):
        # Get data
        outcomes = np.array(
            sql.select(db_path=db_path, table=sim_types[i], get_cols=get_cols)
        )

        # Process data
        for seed in range(constants.NR_SIM_MICE):
            # Process first task
            t1_perf[i][seed, :] = process.get_performance(
                outcomes=outcomes[seed, : trial_nrs[0]]
            )
            t1_xp_t[seed, i] = process.get_expert_t(performances=t1_perf[i][seed, :])

            # Process second task
            t2_perf[i][seed, :] = process.get_performance(
                outcomes=outcomes[seed, trial_nrs[0] :]
            )
            t2_xp_t[seed, i] = process.get_expert_t(performances=t2_perf[i][seed, :])

    return t1_trials, t2_trials, t1_xp_t, t2_xp_t, t1_perf, t2_perf


def plot_performance_bars(
    save: bool = True,
    db_path: Path = SAVE_PATH,
    panel="x",
    plot_name: str = "perf_bar",
    plot_dir: Path = PLOT_DIR,
    colors: list | None = None,
    labels: list[str] | None = None,
    verbose: bool = False,
) -> None:
    # Initializations
    n_sims = 2
    colors = colors or [COLOR_NO_OFC, COLOR_WT]

    # Get data
    _, _, _, _, t1_perf, t2_perf = get_data(db_path=db_path)
    learning_perf: list = [None, None]
    reversal_perf: list = [None, None]
    t = int(TRIAL_NRS[0] - constants.PERF_MAV_N / 2.0) - 1
    for i in range(n_sims):
        learning_perf[i] = t1_perf[i][:, t]
        reversal_perf[i] = t2_perf[i][:, t]

    # Init plot
    helper.set_style()
    fig, axs = plt.subplots(
        1, 2, figsize=(0.5 * helper.PANEL_WIDTH, helper.PANEL_HEIGHT)
    )
    fig = cast(plt.Figure, fig)

    # Plot performance bars
    xs = [1, 0]
    for i in range(n_sims):
        y_means = [np.mean(learning_perf[i]), np.mean(reversal_perf[i])]
        y_std_errs = [np.std(learning_perf[i]), np.std(reversal_perf[i])]
        for j, ps in enumerate((learning_perf[i], reversal_perf[i])):
            axs[j].bar(
                xs[i], y_means[j], yerr=y_std_errs[j], width=0.9, color=colors[i]
            )

            n = len(ps)
            if n <= 1:
                sxs = np.array([xs[i]])
            else:
                sxs = np.linspace(xs[i] - 0.2, xs[i] + 0.2, n)
            axs[j].scatter(sxs, ps, **helper.BP_SCATTER)

    # Compute significant difference statistics
    perfs = (learning_perf, reversal_perf)
    sts = ((1, 0), (1, 0), (0, 0), (1, 1))
    ds = ((0, 0), (1, 1), (0, 1), (0, 1))
    data_pairs = [
        (learning_perf[1], learning_perf[0]),
        (reversal_perf[1], reversal_perf[0]),
        (learning_perf[0], reversal_perf[0]),
        (learning_perf[1], reversal_perf[1]),
    ]
    phase_label = ["Learning", "Reversal"]
    labels = labels or ["Intact\nlOFC", "Silenced\nlOFC"]
    pls: list[str] = [label.replace("\n", "") for label in labels]
    pls.reverse()
    n_test = len(data_pairs)
    pvs = [
        statistics.welch_t_bonferroni(
            perfs[ds[i][0]][sts[i][0]],
            perfs[ds[i][1]][sts[i][1]],
            n_tests=n_test,
            outcome="performance",
            group1=f"{pls[sts[i][0]]} {phase_label[ds[i][0]]}",
            group2=f"{pls[sts[i][1]]} {phase_label[ds[i][1]]}",
            verbose=verbose,
        )
        for i in range(n_test)
    ]

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
            axs[0].text(
                sum(xs[i]) / 2.0,
                ys[i],
                s=helper.get_significance(pvs[i]),
                **helper.FONT_SIGNIFICANT,
            )
        else:
            axs[0].text(sum(xs[i]) / 2.0, ys[i] + 2, **helper.FONT_NON_SIGNIFICANT)

    colors = [colors[1], colors[0]]
    x_circle = [-0.0, 0.5]
    circ: list = [None, None]
    for i in range(2):
        axs[i].spines[["right", "top"]].set_visible(False)
        axs[i].set_xlim([-1, 2])
        circ[i] = patches.Ellipse(
            xy=(x_circle[i], 145.0),
            width=2.8,
            height=40,
            facecolor=colors[i],
            clip_on=False,
        )
        axs[i].add_patch(circ[i])
        axs[i].text(x_circle[i], 143, labels[i], ha="center", va="center")
        axs[i].set_ylim([0, y_max])
    axs[1].set_yticks([])
    axs[0].set_yticks([0.0, 100.0])
    axs[0].tick_params(axis="y", which="major", pad=0.5)
    axs[0].spines[["left"]].set_bounds([0, 100])
    axs[1].spines[["left"]].set_visible(False)
    axs[0].set_ylabel("    Perf. (Trial #600)", labelpad=-5, loc="bottom")
    axs[0].set_xticks([0.5], ["Learning"])
    axs[1].set_xticks([0.5], ["Reversal"])
    axs[1].set_zorder(-1)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.1, wspace=0.0)
    fig.subplots_adjust(top=0.85, bottom=0.12)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=plot_dir, plot_name=plot_name)


def panel_e_performance_bars(save: bool = True, verbose: bool = False) -> None:
    plot_performance_bars(
        save=save, panel="e", plot_name="e_performance_bars", verbose=verbose
    )


def plot_cumulative_distribution(
    save: bool = True,
    db_path: Path = SAVE_PATH,
    panel="x",
    plot_name: str = "cum",
    phase_cosmetics: bool = True,
    plot_dir: Path = PLOT_DIR,
    colors: list | None = None,
    labels: list | None = None,
) -> None:
    # Basic initializations
    simulation_types = [OFC_SIM, NO_OFC_SIM]
    colors = colors or [COLOR_WT, COLOR_NO_OFC]
    if phase_cosmetics:
        k_max = 117.3
    else:
        k_max = 100
    rect_lim = [100, k_max]

    # Database and columns containing data
    get_cols = [f"{keys.PERFORMANCE}_{t}" for t in range(sum(TRIAL_NRS))]

    # Plot expert time pyramid
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(0.8 * helper.PANEL_WIDTH, helper.PANEL_HEIGHT),
        height_ratios=TRIAL_NRS,
        width_ratios=[1, 1, 0.2],
    )
    fig = cast(plt.Figure, fig)
    for j in range(len(TRIAL_NRS)):
        n_trials = TRIAL_NRS[j]
        t2_xp_t: list = [None, None]
        bin_width = TRIAL_NRS[1] / 32.0
        n_bins = int(n_trials / bin_width)
        for i in range(len(simulation_types)):
            all_outcomes = np.array(
                sql.select(
                    db_path=db_path, table=simulation_types[i], get_cols=get_cols
                )
            )
            if j == 0:
                outcomes = all_outcomes[:, : TRIAL_NRS[0]]
            else:
                outcomes = all_outcomes[:, TRIAL_NRS[0] :]
            nr_seeds = outcomes.shape[0]
            t2_xp_t[i] = np.zeros((nr_seeds,))
            for seed in range(nr_seeds):
                performances = process.get_performance(outcomes=outcomes[seed, :])
                t2_xp_t[i][seed] = process.get_expert_t(performances=performances)

            # Extract cumulative distribution
            histo = np.histogram(t2_xp_t[i], bins=n_bins, range=(0, n_trials))
            xs = histo[1][1:] - 0.5 * n_trials / n_bins
            cumulative_dist = 100 * np.cumsum(histo[0]) / nr_seeds

            # Plot pyramid
            axs[j, i].barh(
                xs,
                cumulative_dist,
                color=colors[i],
                height=bin_width,
                linewidth=0.5,
                edgecolor="k",
                clip_on=False,
            )

            # Set axes
            axs[j, i].set_xlim([0, 100])
            axs[j, i].set_ylim([0, n_trials])
            axs[j, i].spines[["right", "top"]].set_visible(False)
            axs[j, i].invert_yaxis()

        # More axes cosmetics
        axs[j, 0].invert_xaxis()
        axs[j, 1].set_yticks([])
        axs[0, j].set_xticks([])
        axs[j, 2].axis("off")
        axs[j, 2].invert_yaxis()

        # Plot cosmetics
        if phase_cosmetics:
            rect = patches.Rectangle(
                xy=(0, constants.NR_TRIALS),
                width=rect_lim[j],
                height=TRIAL_NRS[1] - constants.NR_TRIALS,
                linewidth=0,
                facecolor=(0.0, 0.0, 0.0),
                alpha=0.2,
                clip_on=False,
            )
            axs[1, j].add_patch(rect)
    axs[0, 0].set_yticks([1, constants.NR_TRIALS])
    axs[1, 0].set_yticks([1, constants.NR_TRIALS, TRIAL_NRS[1]])
    fig.supylabel(
        "Trials (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.05, y=0.55
    )
    axs[1, 1].plot(
        [-100, k_max], [-140, -140], clip_on=False, **helper.STYLE_RULE_SWITCH
    )
    axs[1, 0].text(50, -50, clip_on=False, **helper.FONT_RULE_SWITCH)
    axs[1, 1].plot([0, 0], [-300, 2], "k", clip_on=False, linewidth=helper.AXIS_WIDTH)

    # Learning phase cosmetics
    if phase_cosmetics:
        norm = Normalize(vmin=0, vmax=1)
        cbar_style = {"fraction": 1.0, "ticks": [], "aspect": 4, "anchor": (0.5, 1)}
        cb = plt.colorbar(
            ScalarMappable(
                norm=norm, cmap=helper.get_performance_cmap(phase="reversal").reversed()
            ),
            ax=axs[1, 2],
            shrink=0.5,
            **cbar_style,
        )
        cb.ax.zorder = -1
        cb.outline.set_visible(False)
        cb = plt.colorbar(
            ScalarMappable(
                norm=norm, cmap=helper.get_performance_cmap(phase="learning").reversed()
            ),
            ax=axs[0, 2],
            **cbar_style,
        )
        cb.ax.zorder = -1
        cb.outline.set_visible(False)
        font = {
            "x": -9.8,
            "fontsize": 4,
            "color": "white",
            "ha": "center",
            "va": "center",
        }
        axs[0, 2].text(y=0.14, s="LN", **font)
        axs[0, 2].text(y=0.9, s="LE", **font)
        axs[1, 2].text(y=0.07, s="RN", **font)
        axs[1, 2].text(y=0.45, s="RE", **font)

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0.0, hspace=0.3)
    fig.subplots_adjust(bottom=0.2)
    if not phase_cosmetics:
        fig.subplots_adjust(right=0.99)
    helper.set_panel_label(label=panel, fig=fig)

    # Last labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("Fraction experts [%]")
    labels = labels or list(SIM_NAMES.values())
    axs[0, 0].text(50, 30, labels[0], ha="center", clip_on=False)
    axs[0, 1].text(50, 30, labels[1], ha="center", clip_on=False)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=plot_dir, plot_name=plot_name)


def panel_f_cumulative_distribution(save: bool = True) -> None:
    plot_cumulative_distribution(
        save=save, panel="f", plot_name="f_expertise_cumulative_distribution"
    )


@dataclass(frozen=True)
class FigureConfig:
    figure_key: str = keys.FIGURE_3
    panel: str = "g"
    plot_name: str = "g_expert_time"
    plot_dir: Path = PLOT_DIR
    trial_nrs: tuple[int, int] = TRIAL_NRS
    panel_width: float = 0.9 * helper.PANEL_WIDTH
    height_ratios: tuple[float, float] = (4.0, 1.0)
    width_ratios: tuple[float, float] = (1.0, 2.0)
    height_scale: float = 1.0
    sim_types: tuple[ids.SimulationType, ...] = (NO_OFC_SIM, OFC_SIM)
    sim_names: dict = field(default_factory=lambda: SIM_NAMES)
    db_paths: list[Path] = field(default_factory=lambda: [SAVE_PATH])
    colors: list[tuple[float, float, float] | str] = field(
        default_factory=lambda: [COLOR_NO_OFC, COLOR_WT]
    )
    labels: list[str] | None = None
    positions: list[list[float]] = field(default_factory=lambda: [[2.0, 1.0]])
    trace_seed_loop: bool = False
    reversal_scatter_threshold: float | None = None
    reversal_scatter_sim_types: set[Any] | None = None
    shade_rect_clip_on: bool = False
    add_fake_ofc_patch: bool = False
    fake_ofc_patch_target: tuple[int, int] | None = None
    fake_ofc_patch_color_index: int | None = None
    rule_switch_text: tuple[float, float] | None = (
        constants.NR_TRIALS - 45 * helper.PANEL_WIDTH / (0.9 * helper.PANEL_WIDTH),
        20,
    )
    rule_switch_line: tuple[float, float, float, float] | None = (0.0, -7.5, 0.0, 2.5)
    legend_axis: tuple[int, int] = (0, 1)
    legend_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(
            edgecolor=(0.2, 0.2, 0.2),
            loc="lower right",
            frameon=False,
            borderpad=0,
            labelspacing=0.3,
        )
    )
    performance_ylabel: bool = True
    expertise_ylabel_kwargs: dict[str, Any] | None = field(
        default_factory=lambda: dict(rotation=0, y=0.1, labelpad=15)
    )
    supxlabel_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(fontsize=helper.FONT_SIZE, x=0.55, y=0.03)
    )
    adjust_figure_kwargs: dict[str, Any] = field(default_factory=lambda: dict(hspace=0))
    subplots_adjust_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(bottom=0.2)
    )
    left_y_ticks: tuple[list[float], list[str]] | None = field(
        default_factory=lambda: (
            [0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.0],
            ["", "10", "", "30", "", "50", "", "70", "", "90", ""],
        )
    )
    bottom_xticks: dict[tuple[int, int], tuple[list[float], list[str]]] = field(
        default_factory=lambda: {
            (1, 0): ([0, 100, 200, 300, 400, 500], ["", "100", "", "", "400", ""]),
            (1, 1): (
                [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                ["", "100", "", "", "400", "", "", "700", "", "", "1000", "", ""],
            ),
        }
    )
    top_right_hide_y_ticks: bool = True
    y_lims_fixed: tuple[float, float] | None = None
    box_y_limits: tuple[float, float] = (0.5, 2.5)


def _mean_performance(perf: list[np.ndarray], trial_count: int) -> np.ndarray:
    perf_mean = np.zeros((len(perf), trial_count))
    for i, sim_perf in enumerate(perf):
        perf_mean[i, :] = np.mean(sim_perf, axis=0)
    return perf_mean


def plot_expertise_violin(
    ax: plt.Axes,
    xp_t: np.ndarray,
    pos: float,
) -> None:
    n = len(xp_t)
    if n == 0:
        return

    parts = ax.violinplot(
        [xp_t],
        positions=[pos],
        widths=0.55,
        vert=False,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor("k")
        body.set_alpha(1)
        body.set_linewidth(helper.AXIS_WIDTH)
    if "cmedians" in parts and parts["cmedians"] is not None:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(helper.AXIS_WIDTH)
    if n <= 1:
        ys = np.array([pos])
    else:
        ys = np.linspace(pos - 0.2, pos + 0.2, n)
    ax.scatter(xp_t, ys, **helper.BP_SCATTER)


def plot_expert_time(
    save: bool = True, fig_config: FigureConfig = FigureConfig()
) -> None:
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(fig_config.panel_width, fig_config.height_scale * helper.PANEL_HEIGHT),
        height_ratios=list(fig_config.height_ratios),
        width_ratios=list(fig_config.width_ratios),
    )
    fig = cast(Figure, fig)
    axs = cast(np.ndarray, axs)

    global_min_y = 0.0
    global_max_y = 0.0
    cached_data: list[dict[str, Any]] = []

    for d, data_path in enumerate(fig_config.db_paths):
        t1_trials, t2_trials, t1_xp_t, t2_xp_t, t1_perf, t2_perf = get_data(
            db_path=data_path,
            sim_types=fig_config.sim_types,
            trial_nrs=fig_config.trial_nrs,
        )
        global_max_y = max(global_max_y, float(np.max(t1_perf)), float(np.max(t2_perf)))
        cached_data.append(
            dict(
                t1_trials=t1_trials,
                t2_trials=t2_trials,
                t1_xp_t=t1_xp_t,
                t2_xp_t=t2_xp_t,
                t1_perf=t1_perf,
                t2_perf=t2_perf,
                t1_perf_mean=_mean_performance(t1_perf, len(t1_trials)),
                t2_perf_mean=_mean_performance(t2_perf, len(t2_trials)),
                positions=fig_config.positions[d],
                color_offset=d * len(fig_config.sim_types),
            )
        )

    if fig_config.y_lims_fixed is not None:
        global_min_y, global_max_y = fig_config.y_lims_fixed

    if fig_config.figure_key == keys.FIGURE_S7:
        rect = patches.Rectangle(
            xy=(0, global_min_y),
            width=100,
            height=global_max_y - global_min_y,
            linewidth=0,
            facecolor=fig_config.colors[2],
            alpha=0.3,
        )
        axs[0, 1].add_patch(rect)

    for data in cached_data:
        t1_trials = data["t1_trials"]
        t2_trials = data["t2_trials"]
        t1_xp_t = data["t1_xp_t"]
        t2_xp_t = data["t2_xp_t"]
        t1_perf = data["t1_perf"]
        t2_perf = data["t2_perf"]
        t1_perf_mean = data["t1_perf_mean"]
        t2_perf_mean = data["t2_perf_mean"]
        positions = data["positions"]
        color_offset = data["color_offset"]

        if fig_config.trace_seed_loop:
            for seed in range(constants.NR_SIM_MICE):
                for i, _sim_type in enumerate(fig_config.sim_types):
                    color = fig_config.colors[color_offset + i]
                    axs[0, 0].plot(
                        t1_trials, t1_perf[i][seed, :], color=color, alpha=0.2
                    )
                    axs[0, 1].plot(
                        t2_trials, t2_perf[i][seed, :], color=color, alpha=0.2
                    )
        else:
            for i, _sim_type in enumerate(fig_config.sim_types):
                color = fig_config.colors[color_offset + i]
                axs[0, 0].plot(t1_trials, t1_perf[i].T, color=color, alpha=0.2)
                axs[0, 1].plot(t2_trials, t2_perf[i].T, color=color, alpha=0.2)

        mean_order = list(range(len(fig_config.sim_types)))
        if fig_config.figure_key == keys.FIGURE_3:
            mean_order = [1, 0]

        for i in mean_order:
            color = fig_config.colors[color_offset + i]
            label = None
            if fig_config.labels is not None:
                label = fig_config.labels[color_offset + i]
            elif len(fig_config.db_paths) == 1:
                label = fig_config.sim_names[fig_config.sim_types[i]]

            axs[0, 0].plot(
                t1_trials,
                t1_perf_mean[i, :],
                color=color,
                label=label if fig_config.figure_key == keys.FIGURE_S7 else None,
                linewidth=1.5 * helper.LINE_WIDTH,
            )
            axs[0, 1].plot(
                t2_trials,
                t2_perf_mean[i, :],
                color=color,
                label=label,
                linewidth=1.5 * helper.LINE_WIDTH,
            )

        rect_style = dict(
            height=1.0, linewidth=0, alpha=0.4, clip_on=fig_config.shade_rect_clip_on
        )
        for i, pos in enumerate(positions):
            color = fig_config.colors[color_offset + i]
            rect = patches.Rectangle(
                xy=(t1_trials[0], pos - 0.5),
                width=t1_trials[-1],
                facecolor=color,
                **rect_style,
            )
            axs[1, 0].add_patch(rect)
            rect = patches.Rectangle(
                xy=(t2_trials[0], pos - 0.5),
                width=t2_trials[-1],
                facecolor=color,
                **rect_style,
            )
            axs[1, 1].add_patch(rect)

            t1_times = t1_xp_t[~np.isnan(t1_xp_t[:, i]), i]
            t2_times = t2_xp_t[~np.isnan(t2_xp_t[:, i]), i]

            plot_expertise_violin(axs[1, 0], t1_times, pos)
            if len(t2_times) != 0:
                plot_expertise_violin(axs[1, 1], t2_times, pos)

        for ax_idx in range(2):
            axs[0, ax_idx].plot(
                [0, fig_config.trial_nrs[ax_idx] - 1],
                [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
                label="Expert threshold"
                if fig_config.figure_key != keys.FIGURE_S8
                else None,
                zorder=0,
                **helper.STYLE_EXPERT_PERF,
            )

    if (
        fig_config.add_fake_ofc_patch
        and fig_config.fake_ofc_patch_target is not None
        and fig_config.fake_ofc_patch_color_index is not None
    ):
        rect = patches.Rectangle(
            xy=(0, 0),
            width=0,
            height=0,
            facecolor=fig_config.colors[fig_config.fake_ofc_patch_color_index],
            alpha=0.3,
            label="Fake SST active",
        )
        axs[fig_config.fake_ofc_patch_target].add_patch(rect)

    if fig_config.rule_switch_line is not None:
        x0, y0, x1, y1 = fig_config.rule_switch_line
        line_x = (
            cached_data[0]["t2_trials"][0]
            if fig_config.figure_key != keys.FIGURE_S7
            else 0
        )
        axs[1, 1].plot(
            [line_x + x0, line_x + x1],
            [y0, y1],
            clip_on=False,
            zorder=10,
            **helper.STYLE_RULE_SWITCH,
        )

    if fig_config.rule_switch_text is not None:
        text_x, text_y = fig_config.rule_switch_text
        axs[0, 0].text(text_x, text_y, rotation=90, **helper.FONT_RULE_SWITCH)

    if fig_config.performance_ylabel:
        axs[0, 0].set_ylabel("Performance")
    if fig_config.expertise_ylabel_kwargs is not None:
        axs[1, 0].set_ylabel(
            "Trials to\nexpertise", **fig_config.expertise_ylabel_kwargs
        )
    fig.supxlabel("Trials (after start/reversal)", **fig_config.supxlabel_kwargs)
    axs[fig_config.legend_axis].legend(**fig_config.legend_kwargs)

    t1_xlim = [cached_data[0]["t1_trials"][0], cached_data[0]["t1_trials"][-1]]
    t2_xlim = [cached_data[0]["t2_trials"][0], cached_data[0]["t2_trials"][-1]]
    if fig_config.figure_key == keys.FIGURE_S7:
        t1_xlim = [0, fig_config.trial_nrs[0]]
        t2_xlim = [0, fig_config.trial_nrs[1]]

    axs[0, 0].set_xlim(t1_xlim)
    axs[1, 0].set_xlim(t1_xlim)
    axs[0, 1].set_xlim(t2_xlim)
    axs[1, 1].set_xlim(t2_xlim)
    axs[0, 0].set_ylim([global_min_y, global_max_y])
    axs[0, 1].set_ylim([global_min_y, global_max_y])
    axs[1, 0].set_ylim(fig_config.box_y_limits)
    axs[1, 1].set_ylim(fig_config.box_y_limits)
    axs[1, 0].invert_yaxis()
    axs[1, 1].invert_yaxis()

    axs[0, 0].spines[["bottom", "right", "top"]].set_visible(False)
    axs[0, 1].spines[["bottom", "left", "right", "top"]].set_visible(False)
    axs[1, 0].spines[["right", "top"]].set_visible(False)
    axs[1, 1].spines[["left", "right", "top"]].set_visible(False)
    axs[0, 0].spines["left"].set_bounds(global_min_y, 100)

    axs[0, 0].set_xticks([])
    axs[0, 1].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].set_yticks([])

    for (row, col), (ticks, labels) in fig_config.bottom_xticks.items():
        axs[row, col].set_xticks(ticks, labels)

    if fig_config.left_y_ticks is not None:
        ticks, labels = fig_config.left_y_ticks
        axs[0, 0].set_yticks(ticks, labels)
    if fig_config.top_right_hide_y_ticks:
        axs[0, 1].set_yticks([])

    helper.adjust_figure(fig=fig, **fig_config.adjust_figure_kwargs)
    fig.subplots_adjust(**fig_config.subplots_adjust_kwargs)
    helper.set_panel_label(label=fig_config.panel, fig=fig)

    save_kwargs: dict[str, Any] = dict(
        save=save, fig=fig, plot_dir=fig_config.plot_dir, plot_name=fig_config.plot_name
    )
    helper.save_or_show(**save_kwargs)


def panel_g_expert_time(save: bool = True) -> None:
    plot_expert_time(save=save)


def scan():
    simulation.scan.find_best_params(
        simulation_type=NO_OFC_SIM,
        task_id=TASK_ID,
        switch_task=True,
        scan_type=ids.ST_FINAL,
    )
    simulation.scan.find_best_params(
        simulation_type=OFC_SIM, task_id=TASK_ID, switch_task=True, scan_type=ids.ST_TD
    )


def run():
    print("Running simulations for Figure 3efg")
    simulate_condition(simulation_type=OFC_SIM)
    simulate_condition(simulation_type=NO_OFC_SIM)


def plot(save: bool = True):
    panel_e_performance_bars(save=save)
    panel_f_cumulative_distribution(save=save)
    panel_g_expert_time(save=save)
