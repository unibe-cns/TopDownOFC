from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import cast
from lifelines import KaplanMeierFitter
from ofcsst.utils import constants, ids, paths, process
from ofcsst.figures import helper, statistics
from ofcsst.figures.main.f3efg_ofc_sst import (
    COLOR_NO_OFC,
    COLOR_WT,
    plot_expertise_violin,
)

MOUSE_TYPES = (ids.MT_OFC_WT, ids.MT_OFC_KO)
LEARNING_RULES = (ids.LR_L, ids.LR_R)
PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs9"


def panel_a_mouse_d(save: bool = True) -> None:
    # Basic initializations
    mouse_types = (ids.MT_OFC_KO, ids.MT_OFC_WT)
    colors = [COLOR_NO_OFC, COLOR_WT]
    n_mt = len(mouse_types)

    # Get data
    t1_perf = [process.get_mouse_performances(mt, ids.LR_L) for mt in mouse_types]
    t2_perf = [process.get_mouse_performances(mt, ids.LR_R) for mt in mouse_types]
    t1_trials = [np.arange(1, 1 + perf.shape[1]) for perf in t1_perf]
    t2_trials = [np.arange(1, 1 + perf.shape[1]) for perf in t2_perf]
    t_max = max(
        max([len(t1t) for t1t in t1_trials]), max([len(t2t) for t2t in t2_trials])
    )
    x_lim = [t1_trials[0][0], t_max]
    t1_xp_t = [
        np.array([process.get_expert_t(p, simulation=False) for p in mt_perf])
        for mt_perf in t1_perf
    ]
    t2_xp_t = [
        np.array([process.get_expert_t(p, simulation=False) for p in mt_perf])
        for mt_perf in t2_perf
    ]
    t1_xp_t = [t1xt[~np.isnan(t1xt)] for t1xt in t1_xp_t]
    t2_xp_t = [t2xt[~np.isnan(t2xt)] for t2xt in t2_xp_t]
    t1_perf_mean = [np.zeros(len(t1t)) for t1t in t1_trials]
    t2_perf_mean = [np.zeros(len(t2t)) for t2t in t2_trials]
    for i in range(n_mt):
        t1_perf_mean[i] = np.nanmean(t1_perf[i], axis=0)
        t2_perf_mean[i] = np.nanmean(t2_perf[i], axis=0)

    # Init plot
    helper.set_style()
    height_scale = 1.0
    max_y0 = max(
        max([np.nanmax(mp) for mp in t1_perf]), max([np.nanmax(mp) for mp in t2_perf])
    )
    min_y0 = -3.0
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(1.5 * helper.PANEL_WIDTH, height_scale * helper.PANEL_HEIGHT),
        height_ratios=[4 * height_scale, 1],
    )
    fig = cast(plt.Figure, fig)

    # Plot all performance traces
    for i in range(n_mt):
        axs[0, 0].plot(t1_trials[i], t1_perf[i].T, color=colors[i], alpha=0.2)
        axs[0, 1].plot(t2_trials[i], t2_perf[i].T, color=colors[i], alpha=0.2)

    # Plot mean performance traces
    for i in [1, 0]:
        axs[0, 0].plot(
            t1_trials[i],
            t1_perf_mean[i],
            color=colors[i],
            linewidth=1.5 * helper.LINE_WIDTH,
        )
        axs[0, 1].plot(
            t2_trials[i],
            t2_perf_mean[i],
            color=colors[i],
            linewidth=1.5 * helper.LINE_WIDTH,
        )

    # Plot some cosmetics
    rect_style = dict(height=1.0, linewidth=0, alpha=0.4)
    rect = patches.Rectangle(
        xy=(t1_trials[0][0], 0.5), width=t_max, facecolor=COLOR_WT, **rect_style
    )  # pyright: ignore[reportArgumentType]
    axs[1, 0].add_patch(rect)
    rect = patches.Rectangle(
        xy=(t1_trials[0][0], 1.5), width=t_max, facecolor=COLOR_NO_OFC, **rect_style
    )  # pyright: ignore[reportArgumentType]
    axs[1, 0].add_patch(rect)
    rect = patches.Rectangle(
        xy=(t2_trials[0][0], 0.5), width=t_max, facecolor=COLOR_WT, **rect_style
    )  # pyright: ignore[reportArgumentType]
    axs[1, 1].add_patch(rect)
    rect = patches.Rectangle(
        xy=(t2_trials[0][0], 1.5), width=t_max, facecolor=COLOR_NO_OFC, **rect_style
    )  # pyright: ignore[reportArgumentType]
    axs[1, 1].add_patch(rect)
    axs[1, 1].plot(
        [t2_trials[0][0], t2_trials[0][0]],
        [-7.5 * height_scale, 2.5],
        clip_on=False,
        zorder=10,
        **helper.STYLE_RULE_SWITCH,
    )
    axs[0, 0].text(
        x_lim[0] + 0.97 * (x_lim[1] - x_lim[0]),
        -0.7,
        rotation=90,
        **helper.FONT_RULE_SWITCH,
    )
    for i in range(2):
        axs[0, i].plot(
            x_lim,
            [constants.EXPERT_D, constants.EXPERT_D],
            label="Expert threshold",
            zorder=0,
            **helper.STYLE_EXPERT_PERF,
        )

    # Plot violin plots of time to reach expertise
    pos = (2, 1)
    for i in range(n_mt):
        plot_expertise_violin(axs[1, 0], t1_xp_t[i], pos[i])
        plot_expertise_violin(axs[1, 1], t2_xp_t[i], pos[i])

    # Legend and labels
    axs[0, 0].set_ylabel("Discriminability (d')")
    axs[1, 0].set_ylabel("Sess. to\nexpertise", rotation=0, y=0.1, labelpad=15)
    fig.supxlabel(
        "Sessions (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.55, y=0.03
    )

    # Set axes
    for i in range(2):
        axs[i, 0].set_xlim(x_lim)
        axs[i, 1].set_xlim(x_lim)
        axs[0, i].set_ylim([min_y0, max_y0])
        axs[1, i].set_ylim([0.5, 2.5])
        axs[1, i].invert_yaxis()
        axs[0, i].set_xticks([])
        axs[i, 1].set_yticks([])
    axs[0, 0].spines[["bottom", "right", "top"]].set_visible(False)
    axs[0, 1].spines[["bottom", "left", "right", "top"]].set_visible(False)
    axs[1, 0].spines[["right", "top"]].set_visible(False)
    axs[1, 1].spines[["left", "right", "top"]].set_visible(False)
    axs[0, 0].spines["left"].set_bounds(min_y0, 4)
    axs[1, 0].set_yticks([])

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0, wspace=0)
    fig.subplots_adjust(bottom=0.2)
    helper.set_panel_label(label="a", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="a_d_traces")


def plot_kaplan_meier(
    perf_a: np.ndarray,
    perf_b: np.ndarray,
    label_a: str,
    label_b: str,
    color_a: tuple[float, float, float],
    color_b: tuple[float, float, float],
    ax: plt.Axes,
    x_max: int,
    simulation: bool = False,
) -> None:
    # Plot cumulative probability of expertise
    t_a, e_a = process.get_events(perf_a, simulation=simulation)
    t_b, e_b = process.get_events(perf_b, simulation=simulation)
    timeline = np.arange(0, x_max + 1)
    kmf_a = KaplanMeierFitter()
    kmf_b = KaplanMeierFitter()
    kmf_a.fit(t_a, event_observed=e_a, label=label_a, timeline=timeline)
    kmf_a.plot_cumulative_density(ax=ax, ci_show=True, color=color_a)
    kmf_b.fit(t_b, event_observed=e_b, label=label_b, timeline=timeline)
    kmf_b.plot_cumulative_density(ax=ax, ci_show=True, color=color_b)
    for coll in ax.collections:
        coll.set_edgecolor("none")
        coll.set_linewidth(0)
    # Cosmetics
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("")


def panel_b_kaplan_meier(save: bool = True, verbose: bool = False) -> None:
    """
    Plot Kaplan–Meier and compute Log-rank and Cox PH
    """

    # Initializations
    conditions = list(range(len(MOUSE_TYPES)))
    phases = list(range(len(LEARNING_RULES)))
    n_phases = len(phases)
    outcomes = ["Learning expertise", "Reversal expertise"]
    labels = ["WT control", "hM4Di lOFC"]

    # Get performance data
    perf_grid: list[list[np.ndarray]] = [
        [np.empty((0, 0), dtype=float) for _ in conditions] for _ in phases
    ]
    for p in phases:
        for c in conditions:
            perf_grid[p][c] = process.get_mouse_performances(
                MOUSE_TYPES[c], LEARNING_RULES[p]
            )

    # Extract the highest number of sessions needed
    max_nr_cols = max(arr.shape[1] for row in perf_grid for arr in row)
    xt_max = max_nr_cols - 1
    if xt_max % 2 == 0:
        xt_max -= 1

    # Plot
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=1, ncols=n_phases, figsize=(1.5 * helper.PANEL_WIDTH, helper.PANEL_HEIGHT)
    )
    for r in range(n_phases):
        plot_kaplan_meier(
            perf_grid[r][0],
            perf_grid[r][1],
            color_a=COLOR_WT,
            color_b=COLOR_NO_OFC,
            label_a=labels[0],
            label_b=labels[1],
            ax=axs[r],
            x_max=max_nr_cols,
        )

        # Statistical tests
        if verbose:
            statistics.logrank_cox_ph(
                perf_grid[r][0],
                perf_grid[r][1],
                label_a=labels[0],
                label_b=labels[1],
                outcome=outcomes[r],
                n_tests=n_phases,
            )
        axs[r].set_xlim((0, max_nr_cols))
        axs[r].set_xticks(list(range(1, xt_max + 1, 2)))
        for line in axs[r].lines:
            line.set_clip_on(False)

    # Cosmetics
    axs[1].plot([0, 0], [-0.1, 1], clip_on=False, zorder=10, **helper.STYLE_RULE_SWITCH)
    axs[0].text(max_nr_cols * 0.97, 0.35, rotation=90, **helper.FONT_RULE_SWITCH)
    axs[0].spines[["right", "top"]].set_visible(False)
    axs[1].spines[["left", "right", "top"]].set_visible(False)
    axs[1].set_yticks([])
    axs[0].set_ylabel(r"P$_\mathrm{KM}$(session$_{\mathrm{expert}}$<session)")
    fig.supxlabel(
        "Sessions (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.5, y=0.03
    )
    axs[0].legend(
        frameon=False, handlelength=1, bbox_to_anchor=(-0.01, 1.08), loc="upper left"
    )
    leg = axs[1].get_legend()
    if leg is not None:
        leg.remove()

    # Finalize plot
    helper.adjust_figure(fig=fig, hspace=0, wspace=0)
    helper.set_panel_label(label="b", fig=fig)
    helper.save_or_show(
        save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="b_kaplan_meier"
    )


def plot(save: bool = True):
    panel_a_mouse_d(save=save)
    panel_b_kaplan_meier(save=save)
