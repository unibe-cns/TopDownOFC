import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context

import ofcsst.utils.process
from ofcsst.utils import ids, paths, sql, constants
from ofcsst.figures import helper
from ofcsst.figures.main.f4g_multicontext import get_path, get_cols, get_table, TABLES, NR_CONTEXTS

PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs8"
SIM_TYPES = [ids.SIM_CONVEX_OFC, ids.SIM_OFC_XAP_SWITCH]


def plot_both_rules(save: bool = True) -> None:

    # Initializations
    simulation_types = [ids.SIM_CONVEX_OFC, ids.SIM_OFC_XAP_SWITCH]
    n_sims = len(simulation_types)
    db_paths = [get_path(simulation_type=s) for s in simulation_types]
    cols = get_cols(n_trials=constants.NR_TRIALS, with_seed=False)
    trial_ids = list(range(NR_CONTEXTS * constants.NR_TRIALS))
    temp_data = [np.zeros((NR_CONTEXTS, constants.NR_SEEDS, constants.NR_TRIALS)) for _ in range(n_sims)]
    seed_performances = [None, None]
    perf_mean = [None, None]

    rows = [None, 1, 1, 2, 2]
    colors = [helper.COLOR_DEFAULT, helper.COLOR_CONTEXT]

    linestyle_original = '-'
    linestyle_reversed = 'dotted'
    linestyles = [None, linestyle_original, linestyle_reversed, linestyle_original, linestyle_reversed]
    gap_colors = [None, None, 'k', None, 'k']

    # Process performances
    for i in range(n_sims):
        for c in range(NR_CONTEXTS):
            outcomes = np.array(sql.select(db_path=db_paths[i], table=get_table(TABLES[0], context=c),
                                           get_cols=cols[TABLES[0]]))
            for seed in range(constants.NR_SEEDS):
                temp_data[i][c, seed, :] = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, :])
        seed_performances[i] = np.concatenate([temp_data[i][c, :, :] for c in range(NR_CONTEXTS)], axis=1)
        perf_mean[i] = np.mean(seed_performances[i], axis=0)

    # Init plot
    n_rows = 3
    helper.set_style()
    fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(2. * helper.PANEL_WIDTH, 2. * helper.PANEL_HEIGHT))

    # Plot performance traces
    for seed in range(constants.NR_SEEDS):
        for i in range(n_sims):
            axs[0].plot(trial_ids, seed_performances[i][seed, :], color=colors[i], alpha=0.1)

    # Plot mean performances and set a bunch of axes
    labels: list = ["Single-context", "Two-contexts"]
    ymax = max([np.max(temp_data[i]) for i in range(len(simulation_types))])
    y_lims = [[0, ymax], [-0.7, 12], [-0.1, 1.5]]
    for i in [1, 0]:
        axs[0].plot(trial_ids, perf_mean[i], color=colors[i], linewidth=1.5 * helper.LINE_WIDTH, label=labels[i])

    # Plot reversal line cosmetics
    axs[0].set_zorder(100)
    axs[2].set_zorder(100)
    xs = [i * constants.NR_TRIALS - 0.5 for i in range(1, NR_CONTEXTS)]
    yrs = [[5, ymax], [-3., y_lims[1][1] * 1.15], [y_lims[2][0], y_lims[2][1]]]
    for row in range(n_rows):
        axs[row].patch.set_alpha(0.)
        axs[row].set_xlim([0, NR_CONTEXTS * constants.NR_TRIALS])
        axs[row].set_ylim(y_lims[row])
        for x in xs:
            axs[row].plot([x, x], yrs[row], "k--", clip_on=False, zorder=2, linewidth=helper.AXIS_WIDTH)

    # Plot traces of SNR and pi weight
    labels: list[list] = [[None for _ in range(len(TABLES))] for _ in range(n_sims)]
    labels[0][-2] = 'Original rule'
    labels[0][-1] = 'Reversed rule'
    for s in range(n_sims):
        for table_id in range(1, len(TABLES)):
            values = np.concatenate([np.array(sql.select(
                db_path=db_paths[s], table=get_table(TABLES[table_id], context=c), get_cols=cols[TABLES[table_id]]
            )) for c in range(NR_CONTEXTS)], axis=1)
            mv = np.mean(values, axis=0)
            stderr = np.std(values, axis=0)
            axs[rows[table_id]].plot(trial_ids, mv, linestyle=linestyles[table_id], color='grey',
                                     label=labels[s][table_id], gapcolor=gap_colors[table_id])
            axs[rows[table_id]].fill_between(trial_ids, mv - stderr, mv + stderr, color=colors[s], alpha=0.3)

    # axs[0, t].plot([0, NR_TRIALS[t]], 2 * [constants.EXPERT_PERFORMANCE], "k--", linewidth=1.)
    axs[0].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[1].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[2].spines[['right', 'top']].set_visible(False)
    axs[0].set_xticks([])
    axs[1].set_xticks([])

    axs[2].set_xticks([constants.NR_TRIALS * i for i in range(1, NR_CONTEXTS)])
    axs[0].spines['left'].set_bounds(0, 100)
    axs[0].set_yticks([0, 100])
    axs[0].set_ylabel("Performance", labelpad=0)
    axs[1].spines['left'].set_bounds(0, 10)
    axs[1].set_yticks([0, 10])
    axs[1].set_ylabel("Relative gain (go)", loc="center", labelpad=0)
    axs[2].spines['left'].set_bounds(y_lims[2][0], 1)
    axs[2].set_yticks([0, 1])
    axs[2].set_ylabel(r"$\pi$ strength", y=0.4, labelpad=0)
    axs[2].set_xlabel("Trials")
    fig.align_ylabels(axs)

    # Plot cosmetics like legends and reversal lines
    axs[0].legend(loc="lower right", framealpha=1, ncol=2)
    axs[2].legend(loc="upper right", framealpha=1, ncol=4)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.1, wspace=0)
    helper.set_panel_label(label="b", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="b_traces", plot_dpi=400, png=True)


def panel_a_traces(save: bool = True) -> None:

    # Initializations
    n_sims = len(SIM_TYPES)
    db_paths = [get_path(simulation_type=s) for s in SIM_TYPES]
    cols = get_cols(n_trials=constants.NR_TRIALS, with_seed=False)
    trial_ids = list(range(NR_CONTEXTS * constants.NR_TRIALS))
    temp_data = [np.zeros((NR_CONTEXTS, constants.NR_SEEDS, constants.NR_TRIALS)) for _ in range(n_sims)]
    seed_performances = [None, None]
    perf_mean = [None, None]
    colors = [helper.COLOR_DEFAULT, helper.COLOR_CONTEXT]

    # Process performances
    for i in range(n_sims):
        for c in range(NR_CONTEXTS):
            outcomes = np.array(sql.select(db_path=db_paths[i], table=get_table(TABLES[0], context=c),
                                           get_cols=cols[TABLES[0]]))
            for seed in range(constants.NR_SEEDS):
                temp_data[i][c, seed, :] = ofcsst.utils.process.get_performance(outcomes=outcomes[seed, :])
        seed_performances[i] = np.concatenate([temp_data[i][c, :, :] for c in range(NR_CONTEXTS)], axis=1)
        perf_mean[i] = np.mean(seed_performances[i], axis=0)

    # Init plot
    n_rows = 4
    helper.set_style()
    with rc_context({'mathtext.fontset': 'cm'}):
        fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(2. * helper.PANEL_WIDTH, 1.5 * helper.PANEL_HEIGHT))

    # Plot performance traces
    axs[0].plot([0, NR_CONTEXTS * constants.NR_TRIALS], [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
                **helper.STYLE_EXPERT_PERF)
    for seed in range(constants.NR_SEEDS):
        for i in range(n_sims):
            axs[0].plot(trial_ids, seed_performances[i][seed, :], color=colors[i], alpha=0.1)

    # Plot mean performances and set a bunch of axes
    labels: list = ["Single context", "Two contexts"]
    y_lims = [[0, 100], [0, 10], [0, 1], [0, 0.5]]
    for i in [1, 0]:
        axs[0].plot(trial_ids, perf_mean[i], color=colors[i], linewidth=1.5 * helper.LINE_WIDTH, label=labels[i])

    # Plot reversal line cosmetics
    for row in range(n_rows):
        axs[row].patch.set_alpha(0.)
        axs[row].set_xlim([0, NR_CONTEXTS * constants.NR_TRIALS])
        axs[row].set_ylim(y_lims[row])
    for i in range(1, NR_CONTEXTS):
        x = i * constants.NR_TRIALS - 0.5
        axs[3].plot([x, x], [0, y_lims[3][1] * 4.5], clip_on=False, zorder=3, **helper.STYLE_RULE_SWITCH)
        axs[0].text(i * constants.NR_TRIALS - 38, 35, rotation=90, **helper.FONT_RULE_SWITCH)

    # Plot traces of SNR, pi weight and lOFC activity
    for s in range(n_sims):
        for dt in range(2):
            values = np.concatenate([np.array(sql.select(
                db_path=db_paths[s], table=get_table(TABLES[1 + 2 * dt + c % 2], context=c),
                get_cols=cols[TABLES[1 + 2 * dt + c % 2]]
            )) for c in range(NR_CONTEXTS)], axis=1)
            mv = np.mean(values, axis=0)
            axs[dt + 1].plot(trial_ids, mv, color=colors[s])
        values = np.concatenate([np.array(sql.select(
            db_path=db_paths[s], table=get_table(TABLES[5], context=c), get_cols=cols[TABLES[5]]
        )) for c in range(NR_CONTEXTS)], axis=1)
        axs[3].plot(trial_ids, np.mean(values, axis=0), color=colors[s], label=labels[s])

    # Cosmetics
    for a in range(3):
        axs[a].set_xticks([])
        axs[a].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[3].spines[['right', 'top']].set_visible(False)
    axs[3].set_xticks([constants.NR_TRIALS * i for i in range(1, NR_CONTEXTS)], [str(i) for i in range(1, NR_CONTEXTS)])
    axs[3].set_xlabel("Reversal")
    y_labels = ['Performance', r'Rel. gain',
                r'$\pi$-weight', 'lOFC']
    for a in range(4):
        axs[a].set_yticks(y_lims[a])
        axs[a].set_ylabel(y_labels[a], labelpad=2, va='baseline')
    fig.align_ylabels(axs)
    axs[0].legend(bbox_to_anchor=(0.58, 1.), frameon=False, borderpad=0, ncol=2)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.25, wspace=0)
    fig.subplots_adjust(bottom=0.13, right=0.98)
    helper.set_panel_label(label="a", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="a_traces", plot_dpi=400)


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
    fig, axs = plt.subplots(nrows=n_rows, ncols=1, figsize=(helper.PANEL_WIDTH, 1.5 * helper.PANEL_HEIGHT))

    # Plot lOFC activity bars
    phase_length = 100
    cols = [cols[:phase_length], cols[-phase_length:]]
    xs = np.arange(0, 3 * NR_CONTEXTS, 3)
    for r in range(n_rows):
        for s in range(n_sims):
            y_means = np.zeros((NR_CONTEXTS,))
            y_stds = np.zeros((NR_CONTEXTS,))
            for c in range(NR_CONTEXTS):
                ofc = np.mean(np.array(sql.select(db_path=db_paths[s], table=get_table(TABLES[5], context=c),
                                                  get_cols=cols[r])), axis=1)
                y_means[c] = np.mean(ofc)
                y_stds[c] = np.std(ofc) / np.sqrt(constants.NR_SEEDS)
            axs[r].bar(xs + s, y_means, yerr=y_stds, width=0.9, color=colors[s], label=labels[s])

    # Cosmetics
    y_lims = [0, 0.15]
    y_labels = ['lOFC activity (naïve)', 'lOFC activity (expert)']
    for a in range(n_rows):
        axs[a].set_ylim(y_lims)
        axs[a].set_yticks([0, 0.05, 0.1, 0.15])
        axs[a].set_ylabel(y_labels[a])
        axs[a].spines[['right', 'top']].set_visible(False)
        axs[a].set_xticks([c * 3 + 0.5 for c in range(NR_CONTEXTS)], [str(c) for c in range(NR_CONTEXTS)])
    axs[1].set_xlabel("Reversal")
    axs[1].legend(loc='upper right', frameon=False)

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0.25)
    fig.subplots_adjust(left=0.22, bottom=0.13)
    helper.set_panel_label(label="b", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="b_ofc_bars", plot_dpi=400, png=True)


def plot(save: bool = True):
    panel_a_traces(save=save)
    panel_b_ofc_bars(save=save)
