import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from ofcsst.utils import paths
from ofcsst.figures import helper
from ofcsst.figures.main.f2cef_reversal import FakeLearner, init_panel, N_TRIALS


def plot(save: bool = True):

    # Init panel
    helper.set_style()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))
    nr_reversals, nr_trials_per_task, _, trials, x_ticks, x_tick_labels = init_panel()

    # Plot correct action probability
    sim = FakeLearner(nrtrials=nr_trials_per_task)
    perfs = [sim.perf_fct(t) for t in range(nr_trials_per_task)] + [sim.rev_perf_fct(t) for t in
                                                                    range(nr_trials_per_task)]
    axs.plot(trials, perfs, "gray", linewidth=helper.LINE_WIDTH)

    # Learning phase cosmetics
    cmap = helper.get_performance_cmap(phase='both')
    normalize = mcolors.Normalize(vmin=0, vmax=1)
    scalar_map = ScalarMappable(norm=normalize, cmap=cmap)
    cb = plt.colorbar(scalar_map, ax=axs, orientation='horizontal', location='top', ticks=[], drawedges=False,
                      fraction=0.2, aspect=16, pad=0)
    cb.outline.set_visible(False)
    cb.ax.zorder = -1
    x, y = 30, 1.03
    naive_expert_style = {'ha': 'center', 'color': 'white'}
    axs.text(x, y, 'Naïve', **naive_expert_style)
    axs.text(N_TRIALS - x, y, 'Expert', **naive_expert_style)
    axs.text(N_TRIALS + x, y, 'Naïve', **naive_expert_style)
    axs.text(2 * N_TRIALS - x, y, 'Expert', **naive_expert_style)
    ne_x, ne_y = 28,  1.12
    axs.text(ne_x, ne_y, 'LN', ha='center', color=helper.COLOR_LN, weight='bold')
    axs.text(N_TRIALS - ne_x, ne_y, 'LE', ha='center', color=helper.COLOR_LE, weight='bold')
    axs.text(N_TRIALS + ne_x, ne_y, 'RN', ha='center', color=helper.COLOR_RN, weight='bold')
    axs.text(2 * N_TRIALS - ne_x, ne_y, 'RE', ha='center', color=helper.COLOR_RE, weight='bold')

    # Axis cosmetics
    axs.text(N_TRIALS * 0.96, 0.4, rotation=90, **helper.FONT_RULE_SWITCH)
    axs.plot([N_TRIALS, N_TRIALS], [0, 1.2], clip_on=False, **helper.STYLE_RULE_SWITCH)
    axs.set_xlim([0, nr_trials_per_task * nr_reversals])
    axs.set_yticks([0., 1.])
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)
    axs.set_ylabel(r'$\mathregular{p(a_{opt})}$')
    axs.spines.left.set_bounds([0, 1])
    axs.set_ylim([0, 1.])
    axs.set_xticks(x_ticks, x_tick_labels)
    axs.set_xlabel(r'Trial', labelpad=1.7)

    # Final adjustments and annotations
    helper.adjust_figure(fig=fig)
    fig.subplots_adjust(top=0.95)
    helper.set_panel_label(label="d", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=paths.SUPPLEMENTARY_FIG_DIR / "figs1", plot_name="d_prob_opt")
