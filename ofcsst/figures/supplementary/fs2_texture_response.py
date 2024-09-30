import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats
from ofcsst.utils import ids, keys, paths, constants
from ofcsst.figures import helper
from ofcsst.figures.main.f3b_s1_xp import get_path, WITH_OFC


PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs2"
TABLES = [keys.PERFORMANCE, f"{keys.AGENT_GAIN}_T1", f"{keys.AGENT_GAIN}_T2"]
SIM_TYPE = ids.SIM_PG_GAIN
NR_SEEDS = constants.NR_SEEDS
SCAN_TYPE = ids.ST_VARY_DISTRACTOR_1D


def plot(save: bool = True, verbose: bool = False):

    n_phases = 2
    ps = np.arange(start=0, stop=n_phases)
    n_outcomes = 2

    # Get data
    data = np.load(get_path(sim_type=WITH_OFC))
    data = (data[:, :, 0], data[:, :, 1])
    y_max = 105
    y_min_s = 60
    y_min = 60
    y_max_s = 100
    ylabel = 'Sensory response ($|z|$)'
    trial_type = [r'$\mathregular{s_1}$', r'$\mathregular{s_2}$']
    panel = 'b'
    plot_name = 'b_learning_z_responses'

    # Init plot
    helper.set_style()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(0.5 * helper.PANEL_WIDTH, 0.8 * helper.PANEL_HEIGHT))
    outcome_colors = [helper.COLOR_LN, helper.COLOR_LE]
    x_ticks = 3 * ps + 0.5
    x_tick_labels = ['LN', 'LE']
    b_plots = [None, None]
    box_width = 0.85
    bp_props = {'vert': True,
                'widths': box_width,
                'patch_artist': True,
                'showfliers': False,
                'boxprops': dict(linewidth=helper.AXIS_WIDTH),
                'capprops': dict(linewidth=helper.AXIS_WIDTH),
                'whiskerprops': dict(linewidth=helper.AXIS_WIDTH, solid_capstyle='butt'),
                'flierprops': dict(marker='o', markersize=helper.MARKER_SIZE, linewidth=helper.AXIS_WIDTH),
                'medianprops': dict(color='k', solid_capstyle='butt', linewidth=helper.AXIS_WIDTH)}
    for outcome in range(n_outcomes):
        pos = 3 * ps + outcome
        y = [data[outcome][:, i][~np.isnan(data[outcome][:, i])] for i in range(len(pos))]
        b_plots[outcome] = axs.boxplot(x=y, positions=pos, **bp_props)
        for median, patch, color in zip(b_plots[outcome]['medians'], b_plots[outcome]['boxes'], outcome_colors):
            if outcome == 0:
                patch.set(linewidth=0)
                patch.set_facecolor(color)
                if color == outcome_colors[1]:
                    median.set_color('white')
            else:
                patch.set(color=color)
                patch.set_facecolor('white')
    axs.set_ylim([y_min, y_max])
    axs.set_xticks(x_ticks, x_tick_labels)
    axs.set_xlim([-1, 5])
    axs.spines[['right', 'top', 'bottom']].set_visible(False)
    axs.spines.left.set_bounds([y_min_s, y_max_s])
    axs.tick_params(bottom=False)
    tls = axs.get_xticklabels()
    for p in ps:
        tls[p].set_color(outcome_colors[p])
        tls[p].set_fontweight('bold')
    axs.set_ylabel(ylabel)
    hit_patch = patches.Patch(color='black', label=trial_type[0])
    cr_patch = patches.Patch(facecolor='white', ec="black", label=trial_type[1])
    axs.legend(loc="lower right", handles=[hit_patch, cr_patch], handlelength=box_width, frameon=False,
               bbox_to_anchor=(1, 1.), borderpad=0)

    # Significant difference statistics
    phases = ['LN', 'LE', 'RN', 'RE']
    x = 3 * np.arange(start=0, stop=2)
    ymin, y_max = axs.get_ylim()
    y_sig = ymin + 0.9 * (y_max - ymin)
    y_sig_ns = ymin + 0.92 * (y_max - ymin)
    for p in ps:
        d1 = data[0][:, p]
        d2 = data[1][:, p]
        pv = stats.ttest_ind(d1[~np.isnan(d1)], d2[~np.isnan(d2)])[1] * n_phases
        axs.plot([x[p], x[p]+1], [y_sig, y_sig], **helper.STYLE_SIGNIFICANT)
        if pv < constants.SIGNIFICANCE_THRESHOLD:
            axs.text(x=x[p] + 0.5, y=0.99*y_sig_ns, s=helper.get_significance(pv), **helper.FONT_SIGNIFICANT)
        else:
            axs.text(x=x[p] + 0.5, y=y_sig_ns, **helper.FONT_NON_SIGNIFICANT)
        if verbose:
            print(f"{phases[p]}: {trial_type[0]} vs. {trial_type[1]}", pv)

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0.3)
    fig.subplots_adjust(top=0.7)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=plot_name)
