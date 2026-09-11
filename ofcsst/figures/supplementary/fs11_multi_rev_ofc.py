import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
from typing import cast
from ofcsst.utils import ids, paths, sql, constants, process
from ofcsst.figures import helper
from ofcsst.figures.main.f4g_multicontext import (
    get_path,
    get_cols,
    get_table,
    TABLES,
    NR_CONTEXTS,
)

PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs11"
SIM_TYPES: list[ids.SimulationType] = [ids.SIM_CONVEX_OFC, ids.SIM_OFC_XAP_SWITCH]


def panel_a_traces(save: bool = True) -> None:
    # Initializations
    n_sims = len(SIM_TYPES)
    db_paths = [get_path(simulation_type=s) for s in SIM_TYPES]
    cols = get_cols(n_trials=constants.NR_TRIALS, with_seed=False)
    trial_ids = list(range(NR_CONTEXTS * constants.NR_TRIALS))
    temp_data = [
        np.zeros((NR_CONTEXTS, constants.NR_SEEDS, constants.NR_TRIALS))
        for _ in range(n_sims)
    ]
    seed_performances: list = [np.empty(()), np.empty(())]
    perf_mean = [np.empty(()), np.empty(())]
    colors = [helper.COLOR_DEFAULT, helper.COLOR_CONTEXT]

    # Process performances
    for i in range(n_sims):
        for c in range(NR_CONTEXTS):
            outcomes = np.array(
                sql.select(
                    db_path=db_paths[i],
                    table=get_table(TABLES[0], context=c),
                    get_cols=cols[TABLES[0]],
                )
            )
            for seed in range(constants.NR_SEEDS):
                temp_data[i][c, seed, :] = process.get_performance(
                    outcomes=outcomes[seed, :]
                )
        seed_performances[i] = np.concatenate(
            [temp_data[i][c, :, :] for c in range(NR_CONTEXTS)], axis=1
        )
        perf_mean[i] = np.mean(seed_performances[i], axis=0)

    # Init plot
    n_rows = 4
    helper.set_style()
    with rc_context({"mathtext.fontset": "cm"}):
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=1,
            figsize=(2.0 * helper.PANEL_WIDTH, 1.5 * helper.PANEL_HEIGHT),
        )
        fig = cast(plt.Figure, fig)

    # Plot performance traces
    axs[0].plot(
        [0, NR_CONTEXTS * constants.NR_TRIALS],
        [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
        **helper.STYLE_EXPERT_PERF,
    )
    for seed in range(constants.NR_SEEDS):
        for i in range(n_sims):
            axs[0].plot(
                trial_ids, seed_performances[i][seed, :], color=colors[i], alpha=0.1
            )

    # Plot mean performances and set a bunch of axes
    labels: list = ["Single context", "Two contexts"]
    y_lims = [[0, 100], [0, 10], [0, 1], [0, 0.5]]
    for i in [1, 0]:
        axs[0].plot(
            trial_ids,
            perf_mean[i],
            color=colors[i],
            linewidth=1.5 * helper.LINE_WIDTH,
            label=labels[i],
        )

    # Plot reversal line cosmetics
    for row in range(n_rows):
        axs[row].patch.set_alpha(0.0)
        axs[row].set_xlim([0, NR_CONTEXTS * constants.NR_TRIALS])
        axs[row].set_ylim(y_lims[row])
    for i in range(1, NR_CONTEXTS):
        x = i * constants.NR_TRIALS - 0.5
        axs[3].plot(
            [x, x],
            [0, y_lims[3][1] * 4.5],
            clip_on=False,
            zorder=3,
            **helper.STYLE_RULE_SWITCH,
        )
        axs[0].text(
            i * constants.NR_TRIALS - 38, 35, rotation=90, **helper.FONT_RULE_SWITCH
        )

    # Plot traces of SNR, pi weight and lOFC activity
    for s in range(n_sims):
        for dt in range(2):
            values = np.concatenate(
                [
                    np.array(
                        sql.select(
                            db_path=db_paths[s],
                            table=get_table(TABLES[1 + 2 * dt + c % 2], context=c),
                            get_cols=cols[TABLES[1 + 2 * dt + c % 2]],
                        )
                    )
                    for c in range(NR_CONTEXTS)
                ],
                axis=1,
            )
            mv = np.mean(values, axis=0)
            axs[dt + 1].plot(trial_ids, mv, color=colors[s])
        values = np.concatenate(
            [
                np.array(
                    sql.select(
                        db_path=db_paths[s],
                        table=get_table(TABLES[5], context=c),
                        get_cols=cols[TABLES[5]],
                    )
                )
                for c in range(NR_CONTEXTS)
            ],
            axis=1,
        )
        axs[3].plot(
            trial_ids, np.mean(values, axis=0), color=colors[s], label=labels[s]
        )

    # Cosmetics
    for a in range(3):
        axs[a].set_xticks([])
        axs[a].spines[["bottom", "right", "top"]].set_visible(False)
    axs[3].spines[["right", "top"]].set_visible(False)
    axs[3].set_xticks(
        [constants.NR_TRIALS * i for i in range(1, NR_CONTEXTS)],
        [str(i) for i in range(1, NR_CONTEXTS)],
    )
    axs[3].set_xlabel("Reversal")
    y_labels = ["Performance", r"Rel. gain", r"$\pi$-weight", "lOFC"]
    for a in range(4):
        axs[a].set_yticks(y_lims[a])
        axs[a].set_ylabel(y_labels[a], labelpad=2, va="baseline")
    fig.align_ylabels(axs)
    axs[0].legend(bbox_to_anchor=(0.58, 1.0), frameon=False, borderpad=0, ncol=2)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.25, wspace=0)
    fig.subplots_adjust(bottom=0.13, right=0.98)
    helper.set_panel_label(label="a", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="a_traces")


def panel_b_ofc_bars(save: bool = True) -> None:
    # Initializations
    n_sims = len(SIM_TYPES)
    db_paths = [get_path(simulation_type=s) for s in SIM_TYPES]
    cols = get_cols(n_trials=constants.NR_TRIALS, with_seed=False)[TABLES[5]]
    colors = [helper.COLOR_DEFAULT, helper.COLOR_CONTEXT]
    labels: list = ["Single context", "Two contexts"]

    # Init plot
    n_rows = 2
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=1, figsize=(helper.PANEL_WIDTH, 1.5 * helper.PANEL_HEIGHT)
    )
    fig = cast(plt.Figure, fig)

    # Plot lOFC activity bars
    phase_length = 100
    cols = [cols[:phase_length], cols[-phase_length:]]
    xs = np.arange(0, 3 * NR_CONTEXTS, 3)
    for r in range(n_rows):
        for s in range(n_sims):
            y_means = np.zeros((NR_CONTEXTS,))
            y_stds = np.zeros((NR_CONTEXTS,))
            for c in range(NR_CONTEXTS):
                ofc = np.mean(
                    np.array(
                        sql.select(
                            db_path=db_paths[s],
                            table=get_table(TABLES[5], context=c),
                            get_cols=cols[r],
                        )
                    ),
                    axis=1,
                )
                y_means[c] = np.mean(ofc)
                y_stds[c] = np.std(ofc) / np.sqrt(constants.NR_SEEDS)
            axs[r].bar(
                xs + s,
                y_means,
                yerr=y_stds,
                width=0.9,
                color=colors[s],
                label=labels[s],
            )

    # Cosmetics
    y_lims = [0, 0.15]
    y_labels = ["lOFC activity (naïve)", "lOFC activity (expert)"]
    for a in range(n_rows):
        axs[a].set_ylim(y_lims)
        axs[a].set_yticks([0, 0.05, 0.1, 0.15])
        axs[a].set_ylabel(y_labels[a])
        axs[a].spines[["right", "top"]].set_visible(False)
        axs[a].set_xticks(
            [c * 3 + 0.5 for c in range(NR_CONTEXTS)],
            [str(c) for c in range(NR_CONTEXTS)],
        )
    axs[1].set_xlabel("Reversal")
    axs[1].legend(loc="upper right", frameon=False)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.25)
    fig.subplots_adjust(left=0.22, bottom=0.13)
    helper.set_panel_label(label="b", fig=fig)

    # Save or display
    helper.save_or_show(
        save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="b_ofc_bars"
    )


def plot(save: bool = True):
    panel_a_traces(save=save)
    panel_b_ofc_bars(save=save)
