import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from typing import cast, Any

from numpy import ndarray

import ofcsst.utils.process
from ofcsst.utils import constants, ids, keys, paths, process, sql
from ofcsst.figures import helper, statistics
from ofcsst.figures.main.f3efg_ofc_sst import OFC_SIM
from ofcsst.figures.main.f5a_vip_cl import get_path, MOUSE_TYPES, LEARNING_RULES
from ofcsst.figures.supplementary.fs9_ofc_inhibited_perf import plot_kaplan_meier

PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs12"
TRIAL_NRS = (1600, 1600)
TABLES: tuple[str, str] = ("Forward", "Reversed")
COLORS = [helper.COLOR_DEFAULT, helper.COLOR_VIP]
EXP_LABELS = ["WT C57 control", "VIP-Cre/FLEX\nhM4Di in lOFC"]
SIM_LABELS = ["Control", "CL=0"]


def get_linear_fits(
    experimental: bool = True,
) -> tuple[
    list[list[list[Any]]], list[list[list[Any]]], list[list[ndarray[Any, Any]]], int
]:
    # Init fit storage
    ms = [[[], []], [[], []]]
    bs = [[[], []], [[], []]]
    traces: list[list[ndarray[Any, Any]]] = [
        [np.empty(0), np.empty(0)],
        [np.empty(0), np.empty(0)],
    ]
    conditions: list[int] = list(range(len(MOUSE_TYPES)))
    phases: list[int] = list(range(len(LEARNING_RULES)))

    if experimental:
        # Get experimental data
        x_max = 0
        for p in phases:
            for c in conditions:
                data: np.ndarray = process.get_mouse_performances(
                    MOUSE_TYPES[c], LEARNING_RULES[p]
                )
                x_max = max(x_max, data.shape[1])
                for k in range(data.shape[0]):
                    performance = data[k, ~np.isnan(data[k, :])]
                    x = np.linspace(1, len(performance), len(performance))
                    m, b = np.polyfit(x, performance, 1)
                    ms[p][c].append(m)
                    bs[p][c].append(b)
                    data[k, ~np.isnan(data[k, :])] = b + m * x
                traces[p][c] = np.transpose(data)

    else:
        # Get simulated data
        x_max = max(TRIAL_NRS)
        sims: list[ids.SimulationType] = [OFC_SIM, ids.SIM_CONVEX_OFC_VIP_KO]
        for p in phases:
            get_cols = [f"{keys.PERFORMANCE}_{t}" for t in range(TRIAL_NRS[p])]
            for c in conditions:
                data: np.ndarray = np.empty((constants.NR_SEEDS, TRIAL_NRS[p]))
                data[:] = np.nan
                outcomes = np.array(
                    sql.select(
                        db_path=get_path(sim_type=sims[c]),
                        table=TABLES[p],
                        get_cols=get_cols,
                    )
                )
                for seed in range(constants.NR_SEEDS):
                    performances = ofcsst.utils.process.get_performance(
                        outcomes=outcomes[seed, :]
                    )
                    expert_sessions: np.ndarray = (
                        performances > constants.EXPERT_PERFORMANCE
                    )
                    if np.any(expert_sessions):
                        xp_t = min(
                            np.argmax(expert_sessions).item() + 100, TRIAL_NRS[p]
                        )
                    else:
                        xp_t = TRIAL_NRS[p]
                    x = np.linspace(1, xp_t, xp_t)
                    trimmed_perf = performances[:xp_t]
                    m, b = np.polyfit(x, trimmed_perf, 1)
                    ms[p][c].append(m)
                    bs[p][c].append(b)
                    data[seed, :xp_t] = b + m * x
                traces[p][c] = np.transpose(data)

    return ms, bs, traces, x_max


def panel_a_linear_fits(save: bool = True):
    # Init plot
    phase = ["learning", "reversal"]
    phases = list(range(len(phase)))
    conditions = list(range(len(COLORS)))
    plot_rows = [0, 1]
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=len(plot_rows),
        ncols=len(phases),
        figsize=(1.3 * helper.PANEL_WIDTH, 1.2 * helper.PANEL_HEIGHT),
        height_ratios=(19, 16),
    )
    fig = cast(plt.Figure, fig)
    sr, er = 0, 1
    y_lims = [[], []]
    x_maxs = [0, 0]

    # Plot experimental data
    y_lims[er] = [-2, 6.0]
    expert_thresh = 1.5
    xm = [15.9, 17]
    _, _, traces, x_maxs[er] = get_linear_fits(experimental=True)
    for p in phases:
        axs[er, p].plot(
            [1, xm[p]], [expert_thresh, expert_thresh], **helper.STYLE_EXPERT_PERF
        )
        for c in conditions:
            n_x = traces[p][c].shape[0]
            xs = np.linspace(1, n_x, n_x)
            axs[er, p].plot(xs, traces[p][c], color=COLORS[c], alpha=0.2)

    # Plot simulated data fits
    _, _, traces, _ = get_linear_fits(experimental=False)
    y_min = 25
    x_maxs[sr] = max(TRIAL_NRS)
    axs[sr, 1].plot([0, 0], [y_min, 100], clip_on=False, **helper.STYLE_RULE_SWITCH)
    for p in phases:
        axs[sr, p].plot(
            [0, TRIAL_NRS[p]],
            [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
            **helper.STYLE_EXPERT_PERF,
        )
        for c in conditions:
            n_x = traces[p][c].shape[0]
            xs = np.linspace(1, n_x, n_x)
            axs[sr, p].plot(xs, traces[p][c], color=COLORS[c], alpha=0.1)

    y_lims[sr] = list(axs[sr, 1].get_ylim())
    y_lims[sr][0] = y_min
    for p in phases:
        axs[sr, p].set_xlim([0, x_maxs[sr]])
        axs[er, p].set_xlim([1, x_maxs[er]])
        axs[sr, p].set_xticks([500, 1000, 1500])
        axs[er, p].set_xticks([5, 10, 15])
        axs[sr, p].set_ylim(y_lims[sr])
        axs[er, p].set_ylim(y_lims[er])

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

    # Legends
    handles, _ = axs[er, 1].get_legend_handles_labels()
    handles.extend(
        [plt.Line2D([0], [0], label=EXP_LABELS[i], color=COLORS[i]) for i in conditions]
    )
    axs[er, 1].legend(
        handles=handles,
        bbox_to_anchor=(0.36, 0.44),
        borderaxespad=0,
        handlelength=0.7,
        frameon=False,
    )
    handles, _ = axs[sr, 0].get_legend_handles_labels()
    handles.extend(
        [plt.Line2D([0], [0], label=SIM_LABELS[i], color=COLORS[i]) for i in conditions]
    )
    axs[sr, 0].legend(
        handles=handles, loc="lower center", handlelength=0.7, frameon=False
    )

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


def panel_bc_stats(save: bool = True, verbose: bool = False) -> None:
    panel_labels = ("b", "c")
    panel_data = ("slope", "intercept")
    row_ko_condition = ("-VIP", "CL=0")
    row_y_label = ("Experimental\nd'", "Simulated\nperformance")
    row_agent = ("mice", "agents")
    phase_label = ("learning", "reversal")
    condition_colors = ("k", helper.COLOR_VIP)
    conditions: list[int] = list(range(len(condition_colors)))
    n_stat_test = len(phase_label) * len(conditions)
    rs = (1, 0)

    helper.set_style()
    bp_props = {
        "vert": True,
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
    scatter_props = {"s": 2, "alpha": 0.75, "zorder": 10, "lw": 0}
    for panel in range(len(panel_data)):
        fig, axs = plt.subplots(
            nrows=len(row_y_label),
            ncols=len(phase_label),
            figsize=(0.85 * helper.PANEL_WIDTH, 1.2 * helper.PANEL_HEIGHT),
        )
        fig = cast(plt.Figure, fig)
        for row in range(len(row_y_label)):
            r = rs[row]
            experimental = r == 0
            x = get_linear_fits(experimental=experimental)[panel]
            if not isinstance(x, list):
                raise ValueError(f"Expected list, got {type(x)}")
            for p in range(len(phase_label)):
                for c in conditions:
                    axs[row, p].boxplot(positions=[c], x=x[p][c], **bp_props)
                    xx = c + np.linspace(-0.2, 0.2, len(x[p][c]))
                    axs[row, p].scatter(
                        xx, x[p][c], **scatter_props, color=condition_colors[c]
                    )

                axs[row, p].set_xlim([-0.5, 1.5])
                axs[row, p].set_xticklabels(["Control", row_ko_condition[r]])
                axs[row, p].spines[["right", "top"]].set_visible(False)

                # Stats
                pv = statistics.welch_t_bonferroni(
                    x[p][0],
                    x[p][1],
                    n_tests=n_stat_test,
                    outcome=f"{row_agent[r]} {panel_data[panel]} during {phase_label[p]}",
                    group1="Control",
                    group2=row_ko_condition[r],
                    verbose=verbose,
                )
                y_min, y_max = axs[row, p].get_ylim()
                axs[row, p].plot(conditions, [y_max, y_max], "k", lw=helper.AXIS_WIDTH)
                axs[row, p].set_ylim([y_min, y_max + 0.1 * (y_max - y_min)])
                if pv < constants.SIGNIFICANCE_THRESHOLD:
                    axs[row, p].text(
                        sum(conditions) / len(conditions),
                        y_max,
                        s=helper.get_significance(pv),
                        **helper.FONT_SIGNIFICANT,
                    )
                else:
                    axs[row, p].text(
                        sum(conditions) / len(conditions),
                        y_max + 0.02 * (y_max - y_min),
                        **helper.FONT_NON_SIGNIFICANT,
                    )

            axs[row, 0].set_ylabel(f"{row_y_label[r]} {panel_data[panel]}s")

        for p in range(len(phase_label)):
            axs[0, p].set_title(f"{phase_label[p].capitalize()}")

        helper.adjust_figure(fig=fig, hspace=0.4, wspace=0.4)
        fig.subplots_adjust(left=0.26)
        helper.set_panel_label(label=panel_labels[panel], fig=fig)
        helper.save_or_show(
            save=save,
            fig=fig,
            plot_dir=PLOT_DIR,
            plot_name=f"{panel_labels[panel]}_{panel_data[panel]}_stats",
        )


def panel_de_kaplan_meier(save: bool = True, verbose: bool = False) -> None:
    """
    Plot Kaplan–Meier and compute Log-rank and Cox PH
    """

    order = (1, 0)
    simulation = (False, True)
    x_unit = ("Session", "Trial")
    panels = ["d", "e"]
    panels = [panels[i] for i in order]
    labels = (
        [label.replace("\n", " ") for label in EXP_LABELS],
        [label.replace("\n", " ") for label in SIM_LABELS],
    )
    conditions = (0, 1)
    phases = list(range(len(LEARNING_RULES)))
    n_phases = len(phases)
    outcomes = ["Learning expertise", "Reversal expertise"]

    exp_perfs: list[list[np.ndarray]] = [
        [np.empty((0, 0), dtype=float) for _ in MOUSE_TYPES] for _ in phases
    ]
    for p in phases:
        for c in conditions:
            exp_perfs[p][c] = process.get_mouse_performances(
                MOUSE_TYPES[c], LEARNING_RULES[p]
            )

    # Get simulated data
    sims: tuple[ids.SimulationType, ids.SimulationType] = (
        OFC_SIM,
        ids.SIM_CONVEX_OFC_VIP_KO,
    )
    sim_perfs: list[list[np.ndarray]] = [
        [np.empty((constants.NR_SEEDS, TRIAL_NRS[p]), dtype=float) for _ in sims]
        for p in phases
    ]
    for p in phases:
        get_cols = [f"{keys.PERFORMANCE}_{t}" for t in range(TRIAL_NRS[p])]
        for c in conditions:
            data: np.ndarray = np.empty((constants.NR_SEEDS, TRIAL_NRS[p]))
            data[:] = np.nan
            rewards = np.array(
                sql.select(
                    db_path=get_path(sim_type=sims[c]),
                    table=TABLES[p],
                    get_cols=get_cols,
                )
            )
            for seed in range(constants.NR_SEEDS):
                sim_perfs[p][c][seed, :] = ofcsst.utils.process.get_performance(
                    outcomes=rewards[seed, :]
                )
    perfs = (exp_perfs, sim_perfs)

    for i in order:
        max_nr_cols = max(arr.shape[1] for row in perfs[i] for arr in row)
        xt_max = max_nr_cols - 1
        if xt_max % 2 == 0:
            xt_max -= 1
        helper.set_style()
        fig, axs = plt.subplots(
            nrows=1,
            ncols=n_phases,
            figsize=(1.5 * helper.PANEL_WIDTH, helper.PANEL_HEIGHT),
        )
        for r in range(n_phases):
            plot_kaplan_meier(
                perfs[i][r][0],
                perfs[i][r][1],
                color_a=COLORS[0],
                color_b=COLORS[1],
                label_a=labels[i][0],
                label_b=labels[i][1],
                ax=axs[r],
                x_max=max_nr_cols,
                simulation=simulation[i],
            )
            if verbose:
                statistics.logrank_cox_ph(
                    perfs[i][r][0],
                    perfs[i][r][1],
                    label_a=labels[i][0],
                    label_b=labels[i][1],
                    outcome=outcomes[r],
                    n_tests=n_phases,
                    simulation=simulation[i],
                )
            axs[r].set_xlim((0, max_nr_cols))
            if not simulation[i]:
                axs[r].set_xticks(list(range(1, xt_max + 1, 2)))
            else:
                axs[r].set_xticks([500, 1000, 1500])
            for line in axs[r].lines:
                line.set_clip_on(False)

        axs[1].plot(
            [0, 0], [-0.1, 1], clip_on=False, zorder=10, **helper.STYLE_RULE_SWITCH
        )
        axs[0].text(max_nr_cols * 0.97, 0.35, rotation=90, **helper.FONT_RULE_SWITCH)
        axs[0].spines[["right", "top"]].set_visible(False)
        axs[1].spines[["left", "right", "top"]].set_visible(False)
        axs[1].set_yticks([])
        axs[0].set_ylabel(
            r"P$_\mathrm{KM}$("
            + x_unit[i].lower()
            + r"$_{\mathrm{expert}}$<"
            + x_unit[i].lower()
            + r")"
        )
        fig.supxlabel(
            f"{x_unit[i]}s (after start/reversal)",
            fontsize=helper.FONT_SIZE,
            x=0.5,
            y=0.03,
        )
        handles, labs = axs[0].get_legend_handles_labels()
        for ax in axs:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        leg = fig.legend(
            handles,
            labs,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.97),  # figure coordinates
            ncol=len(labels[i]),
            handlelength=1,
        )
        leg.set_zorder(1000)

        helper.adjust_figure(fig=fig, hspace=0, wspace=0)
        fig.subplots_adjust(top=0.86)
        helper.set_panel_label(label=panels[i], fig=fig)
        helper.save_or_show(
            save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=f"{panels[i]}_kaplan_meier"
        )


def plot(save: bool = True) -> None:
    panel_a_linear_fits(save=save)
    panel_bc_stats(save=save)
    panel_de_kaplan_meier(save=save)
